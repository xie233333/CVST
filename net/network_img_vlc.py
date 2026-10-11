import math
from random import choice

import numpy as np
import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ops import quantize_ste as ste_round
from torch.cuda.amp import autocast

from loss.distortion import Distortion
from modules.transform import (
    ChannelContext,
    EntropyParameters,
    HyperAnalysis,
    HyperSynthesis,
    LatentResidualPrediction,
    LocalContext,
)
from net.channel import Channel
from utils_entropy.ckbd import ckbd_anchor, ckbd_merge, ckbd_nonanchor, ckbd_split

from .GDN import GDN

out_channel_F = 64
out_channel_M = 64


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, use_dropout)

    def build_conv_block(self, dim, padding_type, use_dropout):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_layer = nn.Conv2d(dim, dim, kernel_size=3, padding=p)
        torch.nn.init.xavier_normal_(conv_layer.weight.data, math.sqrt(2))
        conv_block += [conv_layer, nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_layer_2 = nn.Conv2d(dim, dim, kernel_size=3, padding=p)
        torch.nn.init.xavier_normal_(conv_layer_2.weight.data, math.sqrt(2))
        conv_block += [conv_layer_2]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        kernel_size=3,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            conv_layer = nn.Conv2d(
                in_channel,
                in_channel,
                kernel_size,
                padding=(kernel_size // 2),
                bias=bias,
            )
            torch.nn.init.xavier_normal_(conv_layer.weight.data, math.sqrt(2))
            torch.nn.init.constant_(conv_layer.bias.data, 0.01)
            m.append(conv_layer)
            if bn:
                m.append(nn.BatchNorm2d(in_channel))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RateAdaptionEncoder(nn.Module):
    def __init__(self, channel_num, rate_choice, mode="CHW"):
        super(RateAdaptionEncoder, self).__init__()
        self.C, self.H, self.W = (channel_num, 16, 16)
        self.rate_num = len(rate_choice)
        self.rate_choice = rate_choice
        self.register_buffer(
            "rate_choice_tensor", torch.tensor(np.asarray(rate_choice))
        )
        self.weight = nn.Parameter(
            torch.zeros(self.rate_num, self.C, max(self.rate_choice))
        )
        self.bias = nn.Parameter(torch.zeros(self.rate_num, max(self.rate_choice)))
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.C)
        torch.nn.init.uniform_(self.bias, -bound, bound)
        # trunc_normal_(self.w, std=.02)
        self.mask = torch.arange(0, max(self.rate_choice)).repeat(
            self.H * self.W, 1
        )  # B, H*W, max(self.rate_choice)

    def forward(self, x, indexes):
        B, C, H, W = x.size()
        x_BLC = x.flatten(2).permute(0, 2, 1)
        if H != self.H or W != self.W:
            self.update_resolution(H, W, x.get_device())
        w = torch.index_select(self.weight, 0, indexes).reshape(B, H * W, self.C, -1)
        b = torch.index_select(self.bias, 0, indexes).reshape(B, H * W, -1)
        mask = self.mask.repeat(B, 1, 1)
        rate_constraint = (
            self.rate_choice_tensor[indexes]
            .reshape(B, H * W, 1)
            .repeat(1, 1, max(self.rate_choice))
        )
        mask[mask < rate_constraint] = 1
        mask[mask != 1] = 0

        x_BLC_masked = (torch.matmul(x_BLC.unsqueeze(2), w).squeeze() + b) * mask
        x_masked = x_BLC_masked.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        mask_BCHW = mask.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x_masked, mask_BCHW

    def update_resolution(self, H, W, device):
        self.num_patches = H * W
        self.mask = torch.arange(0, max(self.rate_choice)).repeat(self.num_patches, 1)
        self.mask = self.mask.to(device)


class RateAdaptionDecoder(nn.Module):
    def __init__(self, channel_num, rate_choice, mode="CHW"):
        super(RateAdaptionDecoder, self).__init__()
        self.C = channel_num
        self.rate_choice = rate_choice
        self.rate_num = len(rate_choice)
        self.weight = nn.Parameter(
            torch.zeros(self.rate_num, max(self.rate_choice), self.C)
        )
        self.bias = nn.Parameter(torch.zeros(self.rate_num, self.C))
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.rate_num)
        torch.nn.init.uniform_(self.bias, -bound, bound)
        # trunc_normal_(self.weight_bias, std=.02)

    def forward(self, x, indexes):
        B, _, H, W = x.size()
        x_BLC = x.flatten(2).permute(0, 2, 1)
        w = torch.index_select(self.weight, 0, indexes).reshape(
            B, H * W, max(self.rate_choice), self.C
        )
        b = torch.index_select(self.bias, 0, indexes).reshape(B, H * W, self.C)
        x_BLC = torch.matmul(x_BLC.unsqueeze(2), w).squeeze() + b  # BLN
        out = x_BLC.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return out


class Channel_encoder(nn.Module):
    def __init__(
        self,
        out_channel_M=out_channel_M,
        rate_choice=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 32, 48, 64],
        C_channel=12,
    ):
        super(Channel_encoder, self).__init__()
        # Resnet 抽取
        self.res1 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.res2 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.snr_attention1 = Snr_attention_module(out_channel_M)
        self.snr_attention2 = Snr_attention_module(out_channel_M)
        self.snr_attention3 = Snr_attention_module(out_channel_M)
        self.ra_encoder = RateAdaptionEncoder(C_channel, rate_choice)
        self.register_buffer(
            "rate_choice_tensor", torch.tensor(np.asarray(rate_choice))
        )
        self.rate_num = len(rate_choice)
        self.rate_token = nn.Parameter(torch.zeros(self.rate_num, C_channel))
        self.refine = Mlp(C_channel * 2, C_channel * 8, C_channel)

        context_window = 5
        slice_num = 4
        slice_ch = out_channel_M // slice_num
        assert slice_ch * slice_num == out_channel_M

        N = out_channel_M
        M = out_channel_M
        self.context_window = context_window
        self.slice_num = slice_num
        self.slice_ch = slice_ch
        self.h_a = HyperAnalysis(M=M, N=N)
        self.h_s = HyperSynthesis(M=M, N=N)
        # Gussian Conditional
        self.gaussian_conditional = GaussianConditional(None)
        self.entropy_bottleneck = EntropyBottleneck(N)

        self.local_context = nn.ModuleList(
            LocalContext(dim=slice_ch) for _ in range(slice_num)
        )

        self.channel_context = nn.ModuleList(
            ChannelContext(in_dim=slice_ch * i, out_dim=slice_ch) if i else None
            for i in range(slice_num)
        )

        self.entropy_parameters_anchor = nn.ModuleList(
            EntropyParameters(in_dim=M * 2 + slice_ch * 4, out_dim=slice_ch * 2)
            if i
            else EntropyParameters(in_dim=M * 2, out_dim=slice_ch * 2)
            for i in range(slice_num)
        )
        self.entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParameters(in_dim=M * 2 + slice_ch * 6, out_dim=slice_ch * 2)
            if i
            else EntropyParameters(in_dim=M * 2 + slice_ch * 2, out_dim=slice_ch * 2)
            for i in range(slice_num)
        )

        # Latent Residual Prediction
        self.lrp_anchor = nn.ModuleList(
            LatentResidualPrediction(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
            for i in range(slice_num)
        )
        self.lrp_nonanchor = nn.ModuleList(
            LatentResidualPrediction(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
            for i in range(slice_num)
        )

    def vlcoding(self, y):
        # stage1：y-->z，并获取z_hat和z_likelihoods
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset
        # Hyper-parameters
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        # stage2：对y分组切片，进行熵编码
        y_slices = y.chunk(self.slice_num, dim=1)  # 16, 4*8, 32, 32
        y_hat_slices = []
        y_likelihoods = []
        for idx, y_slice in enumerate(y_slices):  # 并行化
            slice_anchor, slice_nonanchor = ckbd_split(
                y_slice
            )  # checkboard将特征分为两类 16, 4, 32, 32
            # stage3：对第一个切片进行编码（anchor和nonanchor）
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
                # predict residuals cause by round
                lrp_anchor = self.lrp_anchor[idx](
                    torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1)
                )
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                # local_ctx: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](
                    slice_anchor
                )  # 本地上下文：针对anchor和nonanchor部分的slice（仅本个idx的）
                params_nonanchor = self.entropy_parameters_nonanchor[idx](
                    torch.cat([local_ctx, hyper_params], dim=1)
                )
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(
                    y_slice, scales_slice, means_slice
                )
                # round slice_nonanchor
                slice_nonanchor = (
                    ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
                )
                y_hat_slice = slice_anchor + slice_nonanchor
                # predict residuals cause by round
                lrp_nonanchor = self.lrp_nonanchor[idx](
                    torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1)
                )
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)
            # stage4：对其余切片进行编码（anchor和nonanchor）
            else:
                # global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1)) # 所有之前的slices
                channel_ctx = self.channel_context[idx](
                    torch.cat(y_hat_slices, dim=1)
                )  # 通道上下文：先前熵编码后的slices （以channel区分的）
                # Anchor(Use channel context and hyper params)
                params_anchor = self.entropy_parameters_anchor[
                    idx
                ](
                    torch.cat([channel_ctx, hyper_params], dim=1)
                )  # (torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1)) # 求scales和means时嵌入side info
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
                # predict residuals cause by round
                lrp_anchor = self.lrp_anchor[idx](
                    torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1)
                )
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor(Use spatial context, channel context and hyper params)
                # global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor) # 上一个slice和当前anchor的slice
                # ctx_params: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[
                    idx
                ](
                    torch.cat([local_ctx, channel_ctx, hyper_params], dim=1)
                )  # (torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(
                    y_slice, scales_slice, means_slice
                )
                # round slice_nonanchor
                slice_nonanchor = (
                    ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
                )
                y_hat_slice = slice_anchor + slice_nonanchor
                # predict residuals cause by round
                lrp_nonanchor = self.lrp_nonanchor[idx](
                    torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1)
                )
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)
        # stage5：将获取的所有重建切片组合为y_hat和y_likelihoods
        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)
        return {
            "y_hat": y_hat,
            "y_likelihoods": y_likelihoods,
            "z_likelihoods": z_likelihoods,
        }  # 用NTSCC的rate adaptive encoder/decoder处理

    def forward(self, input_feature, snr):
        # rate adaptation
        B, C, H, W = input_feature.shape
        eta = 0.2
        z = input_feature
        likelihoods_yz = self.vlcoding(z)
        y_likelihoods = likelihoods_yz["y_likelihoods"]
        z_likelihoods = likelihoods_yz["z_likelihoods"]
        y_hat = likelihoods_yz["y_hat"]
        hx = torch.clamp_min(-torch.log(y_likelihoods.detach()) / math.log(2), 0)
        symbol_num = torch.sum(hx, dim=1).flatten(0) * eta
        indexes = torch.searchsorted(self.rate_choice_tensor, symbol_num).clamp(
            0, self.rate_num - 1
        )  # B*H*W 相当于选择具体rate
        # print(indexes)
        z_BLC = z.flatten(2).permute(0, 2, 1)
        px_BLC = y_likelihoods.detach().flatten(2).permute(0, 2, 1)
        z_BLC = z_BLC + self.refine(torch.cat([px_BLC, z_BLC], dim=-1))
        rate_token = torch.index_select(self.rate_token, 0, indexes)  # BL, N
        rate_token = rate_token.reshape(B, H * W, z.shape[1])
        z_BLC = z_BLC + rate_token
        z = z_BLC.reshape(B, H, W, z.shape[1]).permute(0, 3, 1, 2)

        z = self.snr_attention1(self.res1(z), snr)
        z = self.snr_attention2(self.res2(z), snr)
        z = self.snr_attention3(z, snr)

        output_feature, mask = self.ra_encoder(z, indexes)  # 16, 32, 32, 32

        return output_feature, y_hat, mask, indexes, y_likelihoods, z_likelihoods


class Channel_decoder(nn.Module):
    def __init__(
        self,
        out_channel_M=out_channel_M,
        rate_choice=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 32, 48, 64],
        C_channel=12,
    ):
        super(Channel_decoder, self).__init__()
        # Resnet 抽取
        self.res1 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.res2 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.snr_attention1 = Snr_attention_module(out_channel_M)
        self.snr_attention2 = Snr_attention_module(out_channel_M)
        self.snr_attention3 = Snr_attention_module(C_channel)
        self.ra_decoder = RateAdaptionDecoder(C_channel, rate_choice)
        self.rate_num = len(rate_choice)
        self.rate_token = nn.Parameter(torch.zeros(self.rate_num, C_channel))

    def forward(self, input_feature, indexes, snr):
        B, _, H, W = input_feature.shape
        z = self.ra_decoder(input_feature, indexes)

        z_BLC = z.flatten(2).permute(0, 2, 1)
        rate_token = torch.index_select(self.rate_token, 0, indexes)  # BL, N
        rate_token = rate_token.reshape(B, H * W, z.shape[1])
        z_BLC = z_BLC + rate_token
        z = z_BLC.reshape(B, H, W, z.shape[1]).permute(0, 3, 1, 2)

        z = self.snr_attention3(z, snr)
        z = self.snr_attention1(self.res1(z), snr)
        output_feature = self.snr_attention2(self.res2(z), snr)
        return output_feature


class Snr_attention_module(nn.Module):
    def __init__(self, define_ch_num):
        super(Snr_attention_module, self).__init__()
        self.channel_num = define_ch_num
        self.layer1 = nn.Linear(self.channel_num + 1, 16)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(16, self.channel_num)
        self.activation2 = nn.Sigmoid()
        self.spatial_attention = SpatialAttention()

    def forward(self, input_feature, snr):
        (num, ch_num, width, height) = input_feature.shape
        input_mean = torch.mean(
            input_feature.view(input_feature.shape[0], input_feature.shape[1], -1),
            dim=-1,
        )
        input_cat = torch.cat(
            (
                input_mean,
                torch.Tensor([snr for i in range(num)]).reshape(num, 1).cuda(),
            ),
            dim=1,
        )  # 16, 64+1
        factor = self.layer1(input_cat)
        factor = self.activation1(factor)
        factor = self.layer2(factor)
        factor = self.activation2(factor.float())
        result = torch.mul(
            input_feature,
            factor.view(input_feature.shape[0], input_feature.shape[1], 1, 1),
        )
        spatial_attention_factor = self.spatial_attention(result)
        result = torch.mul(result, spatial_attention_factor)
        return result


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, math.sqrt(2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x.float())


class Feature_encoder(nn.Module):
    """
    Encodes video frames to feature domain.
    """

    def __init__(self, latent_channel):
        super(Feature_encoder, self).__init__()
        self.conv_layer1 = self.make_layers(3, [16, 16, 32, 32])
        self.conv_layer2 = self.make_layers(64, [64, 64, 128, 128])
        self.conv_layer3 = self.make_layers(256, [256, 256, 256, 256])

        self.conv1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(64)
        self.conv2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(256)
        self.conv3 = nn.Conv2d(256, latent_channel, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(latent_channel)

    def make_layers(self, in_channels, stem_channels):
        stem_conv = []
        for out_channels in stem_channels:
            m = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
            torch.nn.init.xavier_normal_(m.weight.data, (math.sqrt(2)))
            torch.nn.init.constant_(m.bias.data, 0.01)
            stem_conv.append(m)
            stem_conv.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        stem_conv = nn.Sequential(*stem_conv)
        return stem_conv

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.gdn1(self.conv1(x))
        x = self.conv_layer2(x)
        x = self.gdn2(self.conv2(x))
        x = self.conv_layer3(x)
        x = self.gdn3(self.conv3(x))
        return x


class Feature_decoder(nn.Module):
    """
    Decodes feature domain video frames to pixel domain.
    """

    def __init__(self, in_channel, mid_channel):
        super(Feature_decoder, self).__init__()
        self.deconv_layer1 = self.make_layers(256, [256, 256, 256, 256])
        self.deconv_layer2 = self.make_layers(128, [128, 64, 64, 64])
        self.deconv_layer3 = self.make_layers(32, [32, 16, 16, 3])
        self.igdn1 = GDN(mid_channel, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(
            mid_channel, 256, 3, stride=2, padding=1, output_padding=1
        )
        #         self.deconv2 = nn.ConvTranspose2d(mid_channel, 512, 3, stride=1, padding=1, output_padding=0)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(256, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(
            256, 128, 3, stride=2, padding=1, output_padding=1
        )
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.igdn3 = GDN(64, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(
            64, 32, 3, stride=2, padding=1, output_padding=1
        )
        torch.nn.init.xavier_normal_(
            self.deconv4.weight.data, (math.sqrt(2 * 1 * (3 + 64) / (2 * 64)))
        )
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)

    def make_layers(self, in_channels, stem_channels):
        stem_conv = []
        for out_channels in stem_channels:
            stem_conv.append(nn.ReLU(inplace=True))
            m = nn.ConvTranspose2d(
                in_channels, out_channels, 3, stride=1, padding=1, output_padding=0
            )
            torch.nn.init.xavier_normal_(m.weight.data, (math.sqrt(2)))
            torch.nn.init.constant_(m.bias.data, 0.01)
            stem_conv.append(m)
            in_channels = out_channels

        stem_conv = nn.Sequential(*stem_conv)
        return stem_conv

    def forward(self, x):
        x = self.igdn1(x)
        x = self.deconv2(x)
        x = self.deconv_layer1(x)
        x = self.deconv3(self.igdn2(x))
        x = self.deconv_layer2(x)
        x = self.deconv4(self.igdn3(x))
        x = self.deconv_layer3(x)
        return x


class Img_semantic(nn.Module):
    def __init__(self, args, config):
        super(Img_semantic, self).__init__()
        self.config = config

        self.distortion_loss = Distortion(args)
        self.channel = Channel(args, config)
        self.pass_channel = config.pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction="none")
        self.H = self.W = 0
        self.multiple_snr = args.multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])

        self.downsample = config.downsample
        self.model = args.model

        self.feature_extract = Feature_encoder(out_channel_F)
        self.frame_reconstruct = Feature_decoder(out_channel_F, out_channel_F)
        self.encoder = Channel_encoder(C_channel=64)
        self.decoder = Channel_decoder(C_channel=64)

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(
            x_gen, x_real, normalization=self.config.norm
        )
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def SVD(self, h):
        # h = h.squeeze(0)
        U, S, V = torch.svd(h)

        S_ = torch.diag(S)
        S_ = np.complex64(S_.cpu())
        S_ = torch.from_numpy(S_)

        V_precoder = V  # torch.transpose(torch.conj(V), 0, 1)

        F_receiver = torch.matmul(
            torch.linalg.inv(S_.cuda()), torch.transpose(torch.conj(U), 0, 1)
        )
        V_precoder = V_precoder.cuda()
        F_receiver = F_receiver.cuda()

        return V_precoder, F_receiver

    def stream_demapping(self, input, B):
        input = input.reshape(B, -1)
        return input

    def stream_mapping(self, input, Ntx, B):
        input = input.reshape(B, Ntx, -1)
        return input

    @autocast(dtype=torch.bfloat16)
    def forward(self, input_image, epoch, given_SNR=None):
        h = (torch.randn(8, 8) + 1j * torch.randn(8, 8)) / 1.414
        h = h.to(self.config.device)  # (B, NR, NT)

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        B, _, H, W = input_image.shape
        features = self.feature_extract(input_image)
        features, features_ntc, mask, indexes, y_likelihoods, z_likelihoods = (
            self.encoder(features, chan_param)
        )

        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2) * H * W) / B
        bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2) * H * W) / B
        bpp = bpp_y + bpp_z

        CBR = mask.sum() / input_image.numel() / 2  # float2complex

        # Feature pass channel
        B = features.shape[0]
        # sigma_square = 1.0 / (2 * 10 ** (chan_param / 10))
        features_shape = features.shape
        features = features.reshape(B, -1)
        L = features.shape[1]
        features = features[:, : L // 2] + features[:, L // 2 :] * 1j
        # feature = torch.matmul(V_precoder, feature)
        features = self.stream_mapping(features, 8, B)
        V_precoder, F_receiver = self.SVD(h)
        features = (
            torch.matmul(V_precoder.real, features.real)
            - torch.matmul(V_precoder.imag, features.imag)
            + torch.matmul(V_precoder.imag, features.real) * 1j
            + torch.matmul(V_precoder.real, features.imag) * 1j
        )
        pwr = torch.mean(features.real**2 + features.imag**2) / mask.sum()
        features = features / torch.sqrt(pwr)  # 功率归一化
        noisy_features = self.feature_pass_channel(features, chan_param, h)
        noisy_features = noisy_features * torch.sqrt(pwr)  # 功率去归一化
        # noisy_mv = noisy_mv * torch.conj(h) / (torch.abs(h) ** 2 + sigma_square * 2) # MMSE 均衡
        noisy_features = (
            torch.matmul(F_receiver.real, noisy_features.real)
            - torch.matmul(F_receiver.imag, noisy_features.imag)
            + torch.matmul(F_receiver.imag, noisy_features.real) * 1j
            + torch.matmul(F_receiver.real, noisy_features.imag) * 1j
        )
        noisy_features = self.stream_demapping(noisy_features, B)
        noisy_features = torch.concat(
            [torch.real(noisy_features), torch.imag(noisy_features)], dim=-1
        )
        noisy_features = noisy_features.reshape(features_shape)
        noisy_features = noisy_features * mask

        noisy_features = self.decoder(noisy_features, indexes, chan_param)
        recon_image = self.frame_reconstruct(noisy_features)
        mse = self.squared_difference(
            input_image * 255.0, recon_image.clamp(0.0, 1.0) * 255.0
        )
        loss_G = mse
        recon_image_ntc = self.frame_reconstruct(features_ntc)
        loss_ntc = self.squared_difference(
            input_image * 255.0, recon_image_ntc.clamp(0.0, 1.0) * 255.0
        )

        return (
            recon_image,
            CBR,
            chan_param,
            mse.mean(),
            loss_G.mean(),
            loss_ntc.mean(),
            bpp,
        )

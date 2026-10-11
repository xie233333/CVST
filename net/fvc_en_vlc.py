import math

import numpy as np
import torch
import torch.nn as nn
from mmcv.ops import ModulatedDeformConv2d as DCN
from scipy.optimize import linear_sum_assignment

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ops import quantize_ste as ste_round
from modules.transform import (
    ChannelContext,
    EntropyParameters,
    HyperAnalysis,
    HyperSynthesis,
    LatentResidualPrediction,
    LocalContext,
    MIMOContext,
)
from torch.autograd import Function
from utils_entropy.ckbd import ckbd_anchor, ckbd_merge, ckbd_nonanchor, ckbd_split

from .video_net import flow_warp


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


dim1 = 128
dim2 = 64

out_channel_M = dim2
out_channel_F = dim2
out_channel_O = dim2
patch_num = dim2 // 32  # MIMO N_t num
deform_ks = 3
deform_groups = 8


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
        self.mask = (
            torch.arange(0, max(self.rate_choice)).repeat(self.H * self.W, 1).cuda()
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

        # print(mask.sum()/256/256/3/16)
        # print(mask.sum()/1920/1080/3)
        # rate adaptation
        x_BLC_masked = (torch.matmul(x_BLC.unsqueeze(2), w).squeeze() + b) * mask
        x_masked = x_BLC_masked.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        mask_BCHW = mask.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x_masked, mask_BCHW

    def update_resolution(self, H, W, device):
        self.num_patches = H * W
        self.mask = torch.arange(0, max(self.rate_choice)).repeat(self.num_patches, 1)
        self.mask = self.mask.to(device)


class Snr_attention_module(nn.Module):
    def __init__(self, define_ch_num):
        super(Snr_attention_module, self).__init__()
        self.channel_num = define_ch_num
        self.layer1 = nn.Linear(self.channel_num * 3 + 16 + 1, 64)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(64, self.channel_num)
        self.activation2 = nn.Sigmoid()
        self.spatial_attention = SpatialAttention()

    def forward(self, input_feature, clip_map, snr, qr_A, qr_NA):
        (num, ch_num, width, height) = input_feature.shape
        input_mean = torch.mean(
            input_feature.view(input_feature.shape[0], input_feature.shape[1], -1),
            dim=-1,
        )
        input_cat = torch.cat(
            (
                input_mean,
                torch.Tensor([snr for i in range(num)]).reshape(num, 1).cuda(),
                torch.mean(clip_map, dim=-1).view(num, 16),
                qr_A.squeeze(-1).squeeze(-1).repeat(num, 1),
                qr_NA.squeeze(-1).squeeze(-1).repeat(num, 1),
            ),
            dim=1,
        )  # 16, 64+16
        # print(input_mean[1, 1], torch.mean(clip_map.softmax(dim=-2), dim=-1).view(num, ch_num//4)[1, 1])
        factor = self.layer1(input_cat)
        factor = self.activation1(factor)
        factor = self.layer2(factor)
        factor = self.activation2(factor.float()) * 2
        # print(factor[1])
        result = torch.mul(
            input_feature,
            factor.view(input_feature.shape[0], input_feature.shape[1], 1, 1),
        )
        spatial_attention_factor = self.spatial_attention(result)
        # print(spatial_attention_factor[1].squeeze(0).squeeze(0))
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


class Context_VLC(nn.Module):
    def __init__(self, input_channel, mid_channel, if_context):
        super(Context_VLC, self).__init__()

        context_window = 5
        if if_context:
            slice_num = 4
        else:
            slice_num = 4

        slice_ch = input_channel // slice_num
        assert slice_ch * slice_num == input_channel

        self.if_context = if_context
        N = mid_channel
        M = input_channel
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

        # MIMO Channel Reference based on Channel-context map (B, 2, 8, 8) -> (B, 16, 8) -> (B, 8, 4, 4) 2, 4, 6, 8
        self.MIMO_context = nn.ModuleList(
            MIMOContext(in_dim=2 * (i + 1), out_dim=2 * (i + 1))
            for i in range(slice_num)
        )

        """
        # Global Reference for non-anchors
        self.global_inter_context = nn.ModuleList(
            LinearGlobalInterContext(dim=slice_ch * i, out_dim=slice_ch * 2, num_heads=slice_ch * i // 4) if i else None
            for i in range(slice_num)
        )
        self.global_intra_context = nn.ModuleList(
            LinearGlobalIntraContext(dim=slice_ch) if i else None
            for i in range(slice_num)
        )
        """
        self.entropy_parameters_anchor = nn.ModuleList(
            EntropyParameters(
                in_dim=M * 2 + slice_ch * 4 + (i + 1) * 16, out_dim=slice_ch * 2
            )
            if i
            else EntropyParameters(in_dim=M * 2 + (i + 1) * 16, out_dim=slice_ch * 2)
            for i in range(slice_num)
        )
        self.entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParameters(
                in_dim=M * 2 + slice_ch * 6 + (i + 1) * 16, out_dim=slice_ch * 2
            )
            if i
            else EntropyParameters(
                in_dim=M * 2 + slice_ch * 2 + (i + 1) * 16, out_dim=slice_ch * 2
            )
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

    def forward(self, y, clip_map):
        """
        Using checkerboard context model with mask attention
        which divides y into anchor and non-anchor parts
        non-anchor use anchor as spatial context
        In addition, a channel-wise entropy model is used, too.
        Args:
            x: [B, 3, H, W]
        return:
            x_hat: [B, 3, H, W]
            y_likelihoods: [B, M, H // 16, W // 16]
            z_likelihoods: [B, N, H // 64, W // 64]
            likelihoods: y_likelihoods, z_likelihoods
        """

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
            if self.if_context:
                with torch.no_grad():
                    mimo_ctx = self.MIMO_context[idx](clip_map[:, : (idx + 1) * 2])
            else:
                mimo_ctx = self.MIMO_context[idx](clip_map[:, : (idx + 1) * 2])
            slice_anchor, slice_nonanchor = ckbd_split(
                y_slice
            )  # checkboard将特征分为两类 16, 4, 32, 32
            # stage3：对第一个切片进行编码（anchor和nonanchor）
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](
                    torch.cat([mimo_ctx, hyper_params], dim=1)
                )
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
                    torch.cat([mimo_ctx, local_ctx, hyper_params], dim=1)
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
                    torch.cat([mimo_ctx, channel_ctx, hyper_params], dim=1)
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
                    torch.cat([mimo_ctx, local_ctx, channel_ctx, hyper_params], dim=1)
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


class Channel_frame_encoder(nn.Module):
    def __init__(
        self,
        out_channel_M=dim1,
        rate_choice=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 32, 48, 64],
        C_channel=12,
    ):
        super(Channel_frame_encoder, self).__init__()
        self.res1 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.res2 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.snr_attention1 = Snr_attention_module(out_channel_M)
        self.snr_attention2 = Snr_attention_module(out_channel_M)
        self.snr_attention3 = Snr_attention_module(out_channel_M)
        self.cov = nn.Conv2d(
            out_channel_M * 2, out_channel_M, kernel_size=3, padding=1, stride=1
        )
        self.ra_encoder = RateAdaptionEncoder(C_channel, rate_choice)
        self.diverse_vlcoding = Context_VLC(
            input_channel=C_channel, mid_channel=C_channel, if_context=True
        )
        self.register_buffer(
            "rate_choice_tensor", torch.tensor(np.asarray(rate_choice))
        )
        self.rate_num = len(rate_choice)
        self.rate_token = nn.Parameter(torch.zeros(self.rate_num, C_channel))
        self.refine = Mlp(C_channel * 2, C_channel * 8, C_channel)

    def forward(
        self,
        f_cur,
        context,
        clip_map,
        exp_clip_map,
        snr,
        context_qr_matrix_A,
        context_qr_matrix_NA,
        w,
    ):
        B, C, H, W = f_cur.shape
        """
        if rate == 2:
            eta = 0.2
        elif rate <= 6:
            eta = 0.5
        else:
            eta = 0.8
        """
        eta = (0.3 * w).clamp(0.25, 0.55)
        z = torch.cat((f_cur, context), dim=1)
        z = self.cov(z)
        z_A, z_NA = ckbd_split(z)
        z_A = z_A / context_qr_matrix_A
        z_NA = z_NA / context_qr_matrix_NA
        z = ckbd_merge(z_A, z_NA)
        # rate adaptation
        likelihoods_yz = self.diverse_vlcoding(z, exp_clip_map)
        y_likelihoods = likelihoods_yz["y_likelihoods"]
        z_likelihoods = likelihoods_yz["z_likelihoods"]
        y_hat = likelihoods_yz["y_hat"]
        hx = torch.clamp_min(-torch.log(y_likelihoods.detach()) / math.log(2), 0) * eta
        symbol_num = torch.sum(hx, dim=1).flatten(0)
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

        z = self.snr_attention1(
            self.res1(z), clip_map, snr, context_qr_matrix_A, context_qr_matrix_NA
        )
        z = self.snr_attention2(
            self.res2(z), clip_map, snr, context_qr_matrix_A, context_qr_matrix_NA
        )
        z = self.snr_attention3(
            z, clip_map, snr, context_qr_matrix_A, context_qr_matrix_NA
        )

        output_BCHW, masked_BCHW = self.ra_encoder(z, indexes)  # 16, 32, 32, 32

        return output_BCHW, masked_BCHW, y_hat, indexes, y_likelihoods, z_likelihoods


class Channel_mv_encoder(nn.Module):
    def __init__(
        self,
        out_channel_M=dim1,
        rate_choice=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 32, 48, 64],
        C_channel=12,
    ):
        super(Channel_mv_encoder, self).__init__()
        self.res1 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.res2 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.snr_attention1 = Snr_attention_module(out_channel_M)
        self.snr_attention2 = Snr_attention_module(out_channel_M)
        self.snr_attention3 = Snr_attention_module(C_channel)
        self.ra_encoder = RateAdaptionEncoder(C_channel, rate_choice)
        self.diverse_vlcoding = Context_VLC(
            input_channel=C_channel, mid_channel=C_channel, if_context=False
        )
        self.register_buffer(
            "rate_choice_tensor", torch.tensor(np.asarray(rate_choice))
        )
        self.rate_num = len(rate_choice)
        self.rate_token = nn.Parameter(torch.zeros(self.rate_num, C_channel))
        self.refine = Mlp(C_channel * 2, C_channel * 8, C_channel)

    def forward(
        self,
        input_feature,
        clip_map,
        exp_clip_map,
        snr,
        mv_qr_matrix_A,
        mv_qr_matrix_NA,
        w,
    ):
        B, C, H, W = input_feature.shape
        """
        if rate == 2:
            eta = 0.1
        elif rate <= 6:
            eta = 0.3
        else:
            eta = 0.6
        """
        eta = (0.1 * w).clamp(0.07, 0.15)
        # rate adaptation
        likelihoods_yz = self.diverse_vlcoding(input_feature, exp_clip_map)
        y_likelihoods = likelihoods_yz["y_likelihoods"]
        z_likelihoods = likelihoods_yz["z_likelihoods"]
        y_hat = likelihoods_yz["y_hat"]
        hx = torch.clamp_min(-torch.log(y_likelihoods.detach()) / math.log(2), 0) * eta
        symbol_num = torch.sum(hx, dim=1).flatten(0)
        indexes = torch.searchsorted(self.rate_choice_tensor, symbol_num).clamp(
            0, self.rate_num - 1
        )  # B*H*W 相当于选择具体rate
        # print(indexes)
        z_BLC = input_feature.flatten(2).permute(0, 2, 1)
        px_BLC = y_likelihoods.detach().flatten(2).permute(0, 2, 1)
        z_BLC = z_BLC + self.refine(torch.cat([px_BLC, z_BLC], dim=-1))
        rate_token = torch.index_select(self.rate_token, 0, indexes)  # BL, N
        rate_token = rate_token.reshape(B, H * W, input_feature.shape[1])
        z_BLC = z_BLC + rate_token
        z = z_BLC.reshape(B, H, W, input_feature.shape[1]).permute(0, 3, 1, 2)

        z = self.snr_attention1(
            self.res1(z), clip_map, snr, mv_qr_matrix_A, mv_qr_matrix_NA
        )
        z = self.snr_attention2(
            self.res2(z), clip_map, snr, mv_qr_matrix_A, mv_qr_matrix_NA
        )
        z = self.snr_attention3(z, clip_map, snr, mv_qr_matrix_A, mv_qr_matrix_NA)

        output_BCHW, masked_BCHW = self.ra_encoder(z, indexes)  # 16, 32, 32, 32

        return output_BCHW, masked_BCHW, y_hat, indexes, y_likelihoods, z_likelihoods


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(32, 16, kernel_size=5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=5, padding=2, stride=2)
        self.act = nn.LeakyReLU(0.1)
        self.avg = nn.AdaptiveAvgPool2d((8, 8))
        self.qkv1 = nn.Linear(64, 64 * 3, bias=True)
        self.qkv2 = nn.Linear(64, 64 * 3, bias=True)
        self.lin1 = nn.Linear(128, 512)
        self.lin2 = nn.Linear(512, 512)
        self.scale = nn.Parameter(torch.ones([]) * torch.Tensor([8]))
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.Tensor([1 / 0.07]))
        )
        self.norm = nn.Softmax(dim=-1)

    def encode_context(self, feature):
        # project offset to the uniform dimension
        proj_feature = self.act(self.conv1(feature))
        proj_feature = self.avg(self.act(self.conv2(proj_feature)))
        qkv = self.qkv1(proj_feature.flatten(start_dim=2))
        q, k, v = qkv[:, :, :64], qkv[:, :, 64:128], qkv[:, :, 128:192]  # 16, 8, 64
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        proj_feature = self.norm(attn) @ v
        # print(proj_feature.mean(dim=-1)[1])
        return torch.abs(proj_feature)

    def encode_csi(self, csi):
        proj_csi = self.act(self.lin1(csi))
        proj_csi = self.lin2(proj_csi).view(64, 8)
        # print(proj_csi.mean(dim=0))
        return torch.abs(proj_csi)

    def forward(self, offset, csi):
        # get embedded features and normalize
        offset_features = self.encode_context(offset)
        offset_features = offset_features / (offset_features.norm(dim=-1, keepdim=True))
        csi_features = self.encode_csi(csi)
        # print(csi_features[:,1].squeeze(-1))
        csi_features = csi_features / csi_features.norm(dim=-1, keepdim=True)
        clip_map = []

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        for i in range(0, offset_features.shape[0]):
            clip_m = offset_features[i] @ csi_features * logit_scale
            # print(clip_m.softmax(dim=-2))
            clip_m = clip_m.unsqueeze(0).unsqueeze(
                0
            )  # clip_m.softmax(dim=-1).unsqueeze(0).unsqueeze(0)
            # print(clip_m)
            if i == 0:
                clip_map = clip_m
            else:
                clip_map = torch.cat((clip_map, clip_m), dim=0)
        # print(clip_map.shape)
        return clip_map  # 16, 1, 8, 8


class FVC_ENCODER(nn.Module):
    def __init__(self):
        super().__init__()

        self.motion_estimation = nn.Sequential(
            # nn.Unflatten(dim=0, unflattened_size=(192, 16, 16)),
            nn.Conv2d(dim1, dim2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim2, dim2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim2, dim2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim2, dim2, kernel_size=3, stride=1, padding=1),
        )

        self.motion_compensation = nn.Sequential(
            nn.Conv2d(dim1, dim2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim2, dim2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim2, dim2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim2, dim2, kernel_size=3, stride=1, padding=1),
        )

        self.offset_mask_conv = nn.Conv2d(
            dim2,
            deform_groups * 3 * deform_ks * deform_ks,
            kernel_size=deform_ks,
            stride=1,
            padding=1,
        )

        self.deform_conv = DCN(
            out_channel_F,
            out_channel_F,
            kernel_size=(deform_ks, deform_ks),
            padding=deform_ks // 2,
            deform_groups=deform_groups,
        )

        self.clip = CLIP()
        self.context_encoder = Channel_frame_encoder(out_channel_M=dim2, C_channel=64)
        self.mv_encoder = Channel_mv_encoder(out_channel_M=dim2, C_channel=64)

        self.offset_num = 2
        self.group_num = 4
        self.fusion = nn.Conv2d(
            out_channel_M * self.offset_num, out_channel_M, 1, 1, groups=self.group_num
        )
        self.max_residue_magnitude = 40
        self.conv_offset = nn.Sequential(
            nn.Conv2d(out_channel_M * 3, out_channel_M, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channel_M, out_channel_M, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channel_M, 3 * self.group_num * self.offset_num, 3, 1, 1),
        )

        rate_num = 5
        self.mv_qr_basic_A = nn.Parameter(torch.ones((1, out_channel_F, 1, 1)))
        self.mv_qr_basic_NA = nn.Parameter(torch.ones((1, out_channel_F, 1, 1)))
        self.mv_qr_scale = nn.Parameter(torch.ones((rate_num, 1, 1, 1)))
        self.context_qr_basic_A = nn.Parameter(torch.ones((1, out_channel_F, 1, 1)))
        self.context_qr_basic_NA = nn.Parameter(torch.ones((1, out_channel_F, 1, 1)))
        self.context_qr_scale = nn.Parameter(torch.ones((rate_num, 1, 1, 1)))

    def get_curr_qr(self, qr_basic_A, qr_basic_NA, q_scale):
        qr_basic_A = LowerBound.apply(qr_basic_A, 0.5) * q_scale
        qr_basic_NA = LowerBound.apply(qr_basic_NA, 0.5) * q_scale
        return qr_basic_A, qr_basic_A

    def context_generation(self, f_ref, f_cat):
        B, C, H, W = f_ref.shape
        out = self.conv_offset(f_cat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        mask = torch.sigmoid(mask)  # m_t
        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset.view(B * self.group_num * self.offset_num, 2, H, W)
        mask = mask.view(B * self.group_num * self.offset_num, 1, H, W)
        z = f_ref.repeat(1, self.offset_num, 1, 1)
        z = z.view(B * self.offset_num * self.group_num, C // self.group_num, H, W)
        z = flow_warp(z, offset)
        z = z * mask
        z = z.view(B, C * self.offset_num, H, W)
        context = self.fusion(z)
        return context

    def MIMO_layermapping(self, clip_map):
        _, C, _, _ = clip_map.shape
        w = []

        clip_map = clip_map.mean(dim=0)
        for i in range(C):
            _, arranged_order_i = linear_sum_assignment(
                clip_map[i].cpu().detach().numpy()
            )
            arranged_order_i = torch.from_numpy(arranged_order_i).cuda()
            # print(arranged_order_i)
            w_i = torch.Tensor(
                [
                    clip_map[i][arranged_order_i[j], j]
                    for j in range(len(arranged_order_i))
                ]
            )
            if i == 0:
                w = w_i.unsqueeze(0)
            else:
                w = torch.cat((w, w_i.unsqueeze(0)), dim=0)
        return w

    def forward(
        self, features, ref_features, img_shape, h, snr, mv_q_scales, context_q_scales
    ):
        B, _, H, W = img_shape
        mv_qr_matrix_A, mv_qr_matrix_NA = self.get_curr_qr(
            self.mv_qr_basic_A, self.mv_qr_basic_NA, mv_q_scales
        )
        context_qr_matrix_A, context_qr_matrix_NA = self.get_curr_qr(
            self.context_qr_basic_A, self.context_qr_basic_NA, context_q_scales
        )

        mv, context, clip_map = self.forward_inter(
            features, ref_features, h
        )  # inter_frame
        w = (
            self.MIMO_layermapping(clip_map.detach())
            .cuda()
            .reshape(1, 16, 1, 1)
            .repeat_interleave(4, dim=1)
            * 8
        )
        exp_clip_map = (
            clip_map.permute(0, 3, 1, 2)
            .reshape(B, 8, 4, 4)
            .repeat_interleave(mv.shape[2] // 4, dim=2)
            .repeat_interleave(mv.shape[3] // 4, dim=3)
            .detach()
        )
        mv_A, mv_NA = ckbd_split(mv)
        mv_A = mv_A / mv_qr_matrix_A
        mv_NA = mv_NA / mv_qr_matrix_NA
        mv = ckbd_merge(mv_A, mv_NA)

        mv, mask_mv, m_hat, index_mv, y_likelihoods, z_likelihoods = self.mv_encoder(
            mv, clip_map, exp_clip_map, snr, mv_qr_matrix_A, mv_qr_matrix_NA, w
        )
        bpp_ym = torch.log(y_likelihoods).sum() / (-math.log(2) * H * W) / B
        bpp_zm = torch.log(z_likelihoods).sum() / (-math.log(2) * H * W) / B
        context, mask_context, c_hat, index_context, y_likelihoods, z_likelihoods = (
            self.context_encoder(
                features,
                context,
                clip_map,
                exp_clip_map,
                snr,
                context_qr_matrix_A,
                context_qr_matrix_NA,
                w,
            )
        )
        bpp_yc = torch.log(y_likelihoods).sum() / (-math.log(2) * H * W) / B
        bpp_zc = torch.log(z_likelihoods).sum() / (-math.log(2) * H * W) / B
        bpp = bpp_yc + bpp_ym + bpp_zc + bpp_zm
        mv_res = torch.cat((mv, context), dim=1)
        # print(mask_mv.sum()/256/256/3, mask_context.sum()/256/256/3)
        mask_mv_res = torch.cat((mask_mv, mask_context), dim=1)
        mc_hat = torch.cat((m_hat, c_hat), dim=1)
        return (
            mv_res,
            mask_mv_res,
            mc_hat,
            index_mv,
            index_context,
            bpp,
            clip_map,
            context_qr_matrix_A,
            context_qr_matrix_NA,
            mv_qr_matrix_A,
            mv_qr_matrix_NA,
        )

    def forward_inter(self, f_cur, f_ref, h):
        # feature extraction
        h = h.unsqueeze(0)
        h = torch.cat((h.real, h.imag), dim=0)
        h = h.view(-1)
        clip_map = None
        f_cur = f_cur
        f_ref = f_ref
        f = torch.cat((f_cur, f_ref), dim=1)
        # motion estimation and compensation
        mv = self.motion_estimation(f)
        deformed_f_ref = self.dcn_warp(mv, f_ref)
        f_mc = torch.cat((deformed_f_ref, f_ref), dim=1)
        f_mc = self.motion_compensation(f_mc)
        f_predict = deformed_f_ref + f_mc
        # motion alignment
        f_cat = torch.cat((f_cur, f_predict, mv), dim=1)
        context = self.context_generation(f_ref, f_cat)

        for i in range(0, patch_num):
            clip_map_part = self.clip(
                context.detach()[:, i * 32 : (i + 1) * 32, :, :], h
            )
            if i == 0:
                clip_map = clip_map_part
            else:
                clip_map = torch.cat(
                    (clip_map, clip_map_part), dim=1
                )  # [B, patch_num, embeded_dim, embeded_dim]

        clip_map = clip_map.softmax(dim=-2)

        # print(clip_map[1,1])
        return mv, context, clip_map

    def dcn_warp(self, offset_info, f_ref):
        offset_and_mask = self.offset_mask_conv(offset_info)
        o1, o2, mask = torch.chunk(offset_and_mask, 3, dim=1)
        offset_map = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask.float())
        deformed_f_ref = self.deform_conv(f_ref.float(), offset_map.float(), mask)
        return deformed_f_ref

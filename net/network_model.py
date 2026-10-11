import time
import torch
import functools
from torch import nn
from DVC.mv_estimate import *
from DVC.basics import *
from DVC.GDN import *
from net.channel import Channel
from loss.distortion import Distortion
from random import choice
from torch.cuda.amp import autocast


class Video_semantic(nn.Module):
    def __init__(self, args, config):
        super(Video_semantic, self).__init__()
        self.config = config
        self.mv_est_net = ME_Spynet()
        self.semantic_encoder = Semantic_encoder()
        self.channel_encoder = Channel_encoder()
        self.channel = Channel(args, config)
        self.channel_decoder = Channel_decoder()
        self.semantic_decoder = Semantic_decoder()
        self.res_information_repair = Semantic_restoration(in_channel=3)
        self.squared_difference = torch.nn.MSELoss(reduction="none")
        self.distortion_loss = Distortion(args)

        self.transmit_warp_net = Warp_net()
        self.receive_warp_net = Warp_net()
        self.multiple_snr = args.multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def SVD(self, h):
        # h = h.squeeze(0)
        U, S, V = torch.svd(h)
        inv_S_ = torch.diag(1 / S)

        V_precoder = V  # torch.transpose(torch.conj(V), 0, 1)
        U_conj = torch.conj(U).T
        F_receiver = (
            torch.matmul(inv_S_, U_conj.real) + torch.matmul(inv_S_, U_conj.imag) * 1j
        )
        V_precoder = V_precoder
        F_receiver = F_receiver

        return V_precoder, F_receiver

    def stream_demapping(self, input, B):
        input = input.reshape(B, -1)
        return input

    def stream_mapping(self, input, Ntx, B):
        input = input.reshape(B, Ntx, -1)
        return input

    @autocast()
    def forward(self, input_image, ref_image, epoch, given_SNR=None):
        h = torch.randn(8, 8) + 1j * torch.randn(8, 8)
        h = h.to(self.config.device)  # (B, NR, NT)

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        mv = self.mv_est_net(input_image, ref_image)
        transmit_predict_frame, _ = self.transmit_motion_compensation(
            ref_image, mv
        )  # 参考帧为前一帧
        res_infor = input_image - transmit_predict_frame
        transmission_infor = torch.cat([mv, res_infor], dim=1)
        semantic = self.semantic_encoder(transmission_infor)
        res = self.channel_encoder(semantic, chan_param)
        CBR = (res.numel()) / input_image.numel() / 2

        B = res.shape[0]
        res_shape = res.shape
        res = res.view(B, -1)
        # Feature pass channel
        L = res.shape[1]
        res = res[:, : L // 2] + res[:, L // 2 :] * 1j
        # sigma_square = 1.0 / (2 * 10 ** (chan_param / 10))
        res = self.stream_mapping(res, 8, B)
        V_precoder, F_receiver = self.SVD(h)
        res = (
            torch.matmul(V_precoder.real, res.real)
            - torch.matmul(V_precoder.imag, res.imag)
            + torch.matmul(V_precoder.imag, res.real) * 1j
            + torch.matmul(V_precoder.real, res.imag) * 1j
        )
        pwr = torch.mean(res.real**2 + res.imag**2)
        res = res / torch.sqrt(pwr)  # 功率归一化
        noisy_res = self.feature_pass_channel(res, chan_param, h)
        noisy_res = noisy_res * torch.sqrt(pwr)  # 功率去归一化
        noisy_res = (
            torch.matmul(F_receiver.real, noisy_res.real)
            - torch.matmul(F_receiver.imag, noisy_res.imag)
            + torch.matmul(F_receiver.imag, noisy_res.real) * 1j
            + torch.matmul(F_receiver.real, noisy_res.imag) * 1j
        )
        noisy_res = self.stream_demapping(noisy_res, B)
        noisy_res = torch.concat([torch.real(noisy_res), torch.imag(noisy_res)], dim=-1)
        noisy_res = noisy_res.view(res_shape)

        receive_semantic_feature = self.channel_decoder(noisy_res, chan_param)
        receive_feature = self.semantic_decoder(receive_semantic_feature)
        out_res_infor = receive_feature[:, 2:, :, :]
        out_mv = receive_feature[:, :2, :, :]
        out_res_infor = self.res_information_repair(out_res_infor)

        r, _ = self.receive_motion_compensation(ref_image, out_mv)
        recon_image = r + out_res_infor

        mse = self.squared_difference(
            input_image * 255.0, recon_image.clamp(0.0, 1.0) * 255.0
        )
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0.0, 1.0))
        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean(), input_image

    def transmit_motion_compensation(self, ref_frame, motion_vector):
        warp_frame = flow_warp(ref_frame, motion_vector)
        input_feature = torch.cat((warp_frame, ref_frame), 1)
        prediction_frame = self.transmit_warp_net(input_feature) + warp_frame
        return prediction_frame, warp_frame

    def receive_motion_compensation(self, ref_frame, motion_vector):
        warp_frame = flow_warp(ref_frame, motion_vector)
        input_feature = torch.cat((warp_frame, ref_frame), 1)
        prediction_frame = self.receive_warp_net(input_feature) + warp_frame
        return prediction_frame, warp_frame


class Semantic_encoder(nn.Module):
    def __init__(self, in_channels=5):
        super(Semantic_encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channel_N, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(
            self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / 6))
        )
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(out_channel_N)
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(out_channel_N)
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(out_channel_N)
        self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)
        # 让输入和输出服从相同的分布，这样就能够避免后面层的激活函数的输出值趋向于0
        torch.nn.init.xavier_normal_(
            self.conv4.weight.data,
            (
                math.sqrt(
                    2
                    * (out_channel_M + out_channel_N)
                    / (out_channel_N + out_channel_N)
                )
            ),
        )
        # torch.nn.init.constant_(tensor, val) 用值val填充向量
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        x = self.conv4(x)
        return x


class Semantic_decoder(nn.Module):
    def __init__(self, input_channel=out_channel_M, out_channels=5):
        super(Semantic_decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(
            input_channel, out_channel_N, 3, stride=1, padding=1, output_padding=0
        )
        torch.nn.init.xavier_normal_(
            self.deconv1.weight.data,
            (
                math.sqrt(
                    2
                    * 1
                    * (out_channel_M + out_channel_N)
                    / (out_channel_M + out_channel_M)
                )
            ),
        )
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(out_channel_N, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(
            out_channel_N, out_channel_N, 3, stride=2, padding=1, output_padding=1
        )
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(out_channel_N, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(
            out_channel_N, out_channel_N, 3, stride=2, padding=1, output_padding=1
        )
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.igdn3 = GDN(out_channel_N, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(
            out_channel_N, out_channels, 3, stride=2, padding=1, output_padding=1
        )
        torch.nn.init.xavier_normal_(
            self.deconv4.weight.data,
            (math.sqrt(2 * 1 * (out_channel_N + 3) / (out_channel_N + out_channel_N))),
        )
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)

    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        x = self.deconv4(x)
        return x


class Channel_encoder(nn.Module):
    def __init__(self, C_channel=12):
        super(Channel_encoder, self).__init__()
        activation = nn.ReLU(True)
        # Resnet 抽取
        self.res1 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.res2 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.snr_attention1 = Snr_attention_module(out_channel_M)
        self.snr_attention2 = Snr_attention_module(out_channel_M)
        self.projection = nn.Conv2d(
            out_channel_M, C_channel, kernel_size=3, padding=1, stride=1
        )

    def forward(self, input_feature, snr):
        z = self.snr_attention1(self.res1(input_feature), snr)
        z = self.snr_attention2(self.res2(z), snr)
        output_feature = self.projection(z)
        return output_feature


class Channel_decoder(nn.Module):
    def __init__(self, input_channel=12):
        super(Channel_decoder, self).__init__()
        # Resnet 抽取
        self.projection_transpose = nn.ConvTranspose2d(
            input_channel, out_channel_M, 3, stride=1, padding=1, output_padding=0
        )
        self.res1 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.res2 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.snr_attention1 = Snr_attention_module(out_channel_M)
        self.snr_attention2 = Snr_attention_module(out_channel_M)

    def forward(self, input_feature, snr):
        z = self.projection_transpose(input_feature)
        z = self.snr_attention1(self.res1(z), snr)
        output_feature = self.snr_attention2(self.res2(z), snr)
        return output_feature


class Snr_attention_module(nn.Module):
    def __init__(self, define_ch_num):
        super(Snr_attention_module, self).__init__()
        self.channel_num = define_ch_num
        self.layer1 = nn.Linear(self.channel_num + 1, self.channel_num // 16)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(self.channel_num // 16, self.channel_num)
        self.activation2 = nn.Sigmoid()
        self.spatial_attention = SpatialAttention()

    def forward(self, input_feature, snr):
        (num, ch_num, width, height) = input_feature.shape
        if ch_num != self.channel_num:
            raise Exception("This input feature is illegal!")
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
        )
        factor = self.layer1(input_cat)
        factor = self.activation1(factor)
        factor = self.layer2(factor)
        factor = self.activation2(factor)
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
        return self.sigmoid(x)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, use_dropout):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
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


# semantic repair
class Semantic_restoration(nn.Module):
    def __init__(self, in_channel, kernel_size=3):
        out_channel = in_channel
        super(Semantic_restoration, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.CAB1 = RCAB(32, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.CAB2 = RCAB(64, 64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.CAB3 = RCAB(128, 128)
        self.deconv3 = nn.ConvTranspose2d(
            128, 64, kernel_size, stride=2, padding=1, output_padding=1
        )
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.CAB11 = RCAB(64, 64)
        self.deconv4 = nn.ConvTranspose2d(
            64, 32, kernel_size, stride=2, padding=1, output_padding=1
        )
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)
        self.CAB12 = RCAB(32, 32)
        self.conv6 = nn.Conv2d(32, out_channel, kernel_size, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv6.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv6.bias.data, 0.01)

    def forward(self, input_feature):
        feature = self.CAB1(self.conv1(input_feature))
        feature = self.CAB2(self.conv2(feature))
        feature = self.CAB3(self.conv3(feature))
        feature = self.CAB11(self.deconv3(feature))
        c_feature = self.CAB12(self.deconv4(feature))

        out_frame = self.conv6(c_feature)
        return out_frame


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        conv_layer_1 = nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True)
        torch.nn.init.xavier_normal_(conv_layer_1.weight.data, math.sqrt(2))
        torch.nn.init.constant_(conv_layer_1.bias.data, 0.01)
        conv_layer_2 = nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True)
        torch.nn.init.xavier_normal_(conv_layer_2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(conv_layer_2.bias.data, 0.01)
        self.conv_du = nn.Sequential(
            conv_layer_1, nn.ReLU(inplace=True), conv_layer_2, nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self,
        in_channel,
        inter_channel,
        reduction=1,
        kernel_size=3,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            conv_layer = nn.Conv2d(
                in_channel,
                inter_channel,
                kernel_size,
                padding=(kernel_size // 2),
                bias=bias,
            )
            torch.nn.init.xavier_normal_(conv_layer.weight.data, math.sqrt(2))
            torch.nn.init.constant_(conv_layer.bias.data, 0.01)
            modules_body.append(conv_layer)
            if bn:
                modules_body.append(nn.BatchNorm2d(inter_channel))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(inter_channel, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# Residual block
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

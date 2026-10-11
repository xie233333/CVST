from loss.distortion import Distortion
from net.channel import Channel
from random import choice
import torch.nn as nn
import torch
import numpy as np
import math
from net.fvc_en_vlc import FVC_ENCODER
from net.fvc_de_vlc import FVC_DECODER
from .GDN import *
from torch.cuda.amp import autocast


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


class Channel_encoder(nn.Module):
    def __init__(self, C_channel=16, norm_layer=nn.BatchNorm2d):
        super(Channel_encoder, self).__init__()
        # Resnet 抽取
        self.res1 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.res2 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.snr_attention1 = Snr_attention_module(out_channel_M)
        self.snr_attention2 = Snr_attention_module(out_channel_M)
        self.snr_attention3 = Snr_attention_module(C_channel)
        self.projection = nn.Conv2d(
            out_channel_M, C_channel, kernel_size=3, padding=1, stride=1
        )

    def forward(self, input_feature, snr):
        z = self.snr_attention1(self.res1(input_feature), snr)
        z = self.snr_attention2(self.res2(z), snr)
        output_feature = self.projection(z)
        return output_feature


class Channel_decoder(nn.Module):
    def __init__(self, C_channel=12):
        super(Channel_decoder, self).__init__()
        # Resnet 抽取
        self.projection_transpose = nn.ConvTranspose2d(
            C_channel, out_channel_M, 3, stride=1, padding=1, output_padding=0
        )
        self.res1 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.res2 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.snr_attention1 = Snr_attention_module(out_channel_M)
        self.snr_attention2 = Snr_attention_module(out_channel_M)
        self.snr_attention3 = Snr_attention_module(C_channel)

    def forward(self, input_feature, snr):
        z = self.projection_transpose(self.snr_attention3(input_feature, snr))
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


out_channel_F = 64
out_channel_M = 64


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
        self.encoder = Channel_encoder(C_channel=40)
        self.decoder = Channel_decoder(C_channel=40)

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

    @autocast()  # (dtype=torch.bfloat16)
    def forward(self, input_image, epoch, given_SNR=None):
        h = (torch.randn(8, 8) + 1j * torch.randn(8, 8)) / 1.414
        h = h.to(self.config.device)  # (B, NR, NT)

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        features = self.feature_extract(input_image)
        features = self.encoder(features, chan_param)
        CBR = features.numel() / input_image.numel() / 2  # float2complex

        # Feature pass channel
        B = features.shape[0]
        features_shape = features.shape
        features = features.view(B, -1)
        L = features.shape[1]
        features = features[:, : L // 2] + features[:, L // 2 :] * 1j
        features = self.stream_mapping(features, 8, B)
        V_precoder, F_receiver = self.SVD(h)
        features = (
            torch.matmul(V_precoder.real, features.real)
            - torch.matmul(V_precoder.imag, features.imag)
            + torch.matmul(V_precoder.imag, features.real) * 1j
            + torch.matmul(V_precoder.real, features.imag) * 1j
        )
        pwr = torch.mean(features.real**2 + features.imag**2)
        features = features / torch.sqrt(pwr)  # 功率归一化
        noisy_features = self.feature_pass_channel(features, chan_param, h)
        noisy_features = noisy_features * torch.sqrt(pwr)  # 功率去归一化
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
        noisy_features = noisy_features.view(features_shape)

        noisy_features = self.decoder(noisy_features, chan_param)
        recon_image = self.frame_reconstruct(noisy_features)
        mse = self.squared_difference(
            input_image * 255.0, recon_image.clamp(0.0, 1.0) * 255.0
        )
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0.0, 1.0))

        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean()

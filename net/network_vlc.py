import math
from random import choice

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from loss.distortion import Distortion
from net.channel import Channel
from net.fvc_de_vlc import FVC_DECODER
from net.fvc_en_vlc import FVC_ENCODER

from .GDN import GDN


class FeatureEncoder(nn.Module):
    """Encode video frames into the feature domain."""

    def __init__(self, latent_channel):
        super().__init__()
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


class FeatureDecoder(nn.Module):
    """Decode feature tensors back into pixel-domain frames."""

    def __init__(self, in_channel, mid_channel):
        super().__init__()
        self.deconv_layer1 = self.make_layers(256, [256, 256, 256, 256])
        self.deconv_layer2 = self.make_layers(128, [128, 64, 64, 64])
        self.deconv_layer3 = self.make_layers(32, [32, 16, 16, 3])
        self.igdn1 = GDN(mid_channel, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(
            mid_channel, 256, 3, stride=2, padding=1, output_padding=1
        )
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


# Preserve the original public names for downstream code that imports them.
Feature_encoder = FeatureEncoder
Feature_decoder = FeatureDecoder

FEATURE_CHANNELS = 64


class WITT(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.config = config
        self.args = args

        self.encoder = FVC_ENCODER()
        self.decoder = FVC_DECODER()

        self.distortion_loss = Distortion(args)
        self.info_loss = nn.CrossEntropyLoss()
        self.channel = Channel(args, config)
        self.pass_channel = config.pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction="none")
        self.H = self.W = 0
        self.multiple_rate = config.lamda
        self.multiple_snr = [int(snr) for snr in args.multiple_snr.split(",")]
        self.downsample = config.downsample
        self.model = args.model

        self.feature_extract = FeatureEncoder(FEATURE_CHANNELS)
        self.frame_reconstruct = FeatureDecoder(FEATURE_CHANNELS, FEATURE_CHANNELS)

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(
            x_gen, x_real, normalization=self.config.norm
        )
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def decompose_channel(self, channel_matrix):
        """Return the SVD precoder, receiver, and singular values."""
        left_vectors, singular_values, right_vectors = torch.svd(channel_matrix)
        inverse_singular_values = torch.diag(1 / singular_values)

        conjugate_left_vectors = torch.conj(left_vectors).T
        receiver = (
            torch.matmul(inverse_singular_values, conjugate_left_vectors.real)
            + torch.matmul(inverse_singular_values, conjugate_left_vectors.imag) * 1j
        )

        return right_vectors, receiver, singular_values

    SVD = decompose_channel

    def stream_demapping(self, input, B):
        input = input.reshape(B, -1)
        return input

    def stream_mapping(self, input, Ntx, B):
        input = input.reshape(B, Ntx, -1)
        return input

    @autocast(dtype=torch.bfloat16)
    def forward(
        self,
        input_image,
        ref_image,
        epoch,
        mv_q_scales=None,
        context_q_scales=None,
        given_SNR=None,
    ):
        h = (torch.randn(8, 8) + 1j * torch.randn(8, 8)) / 1.414
        h = h.to(self.config.device)  # (B, NR, NT)
        precoder, receiver, _ = self.decompose_channel(h)
        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        features = self.feature_extract(input_image)
        with torch.no_grad():
            ref_features = self.feature_extract(ref_image)

        (
            mv,
            mask_mv,
            mc_hat,
            index_offset,
            index_context,
            bpp,
            clip_map,
            context_qr_matrix_A,
            context_qr_matrix_NA,
            mv_qr_matrix_A,
            mv_qr_matrix_NA,
        ) = self.encoder(
            features,
            ref_features,
            input_image.shape,
            h,
            chan_param,
            mv_q_scales,
            context_q_scales,
        )

        CBR = mask_mv.sum() / input_image.numel() / 2
        # Feature pass channel
        B = mv.shape[0]
        # mv_res
        mv_shape = mv.shape
        mv = mv.reshape(B, -1)
        L = mv.shape[1]
        mv = mv[:, : L // 2] + mv[:, L // 2 :] * 1j

        mv = self.stream_mapping(mv, 8, B)
        mv = (
            torch.matmul(precoder.real, mv.real)
            - torch.matmul(precoder.imag, mv.imag)
            + torch.matmul(precoder.imag, mv.real) * 1j
            + torch.matmul(precoder.real, mv.imag) * 1j
        )
        pwr = torch.sum(mv.real**2 + mv.imag**2) / mask_mv.sum()
        mv = mv / torch.sqrt(pwr)  # 功率归一化
        noisy_mv = self.feature_pass_channel(mv, chan_param, h)
        noisy_mv = noisy_mv * torch.sqrt(pwr)  # 功率去归一化

        noisy_mv = (
            torch.matmul(receiver.real, noisy_mv.real)
            - torch.matmul(receiver.imag, noisy_mv.imag)
            + torch.matmul(receiver.imag, noisy_mv.real) * 1j
            + torch.matmul(receiver.real, noisy_mv.imag) * 1j
        )
        noisy_mv = self.stream_demapping(noisy_mv, B)
        noisy_mv = torch.concat([torch.real(noisy_mv), torch.imag(noisy_mv)], dim=-1)
        noisy_mv = noisy_mv.reshape(mv_shape)
        mv_hat = noisy_mv * mask_mv

        f_reconstructions = self.decoder(
            ref_features,
            mv_hat,
            index_offset,
            index_context,
            clip_map,
            h,
            chan_param,
            context_qr_matrix_A,
            context_qr_matrix_NA,
            mv_qr_matrix_A,
            mv_qr_matrix_NA,
        )
        recon_image = self.frame_reconstruct(f_reconstructions)
        loss_G = self.squared_difference(
            input_image * 255.0, recon_image.clamp(0.0, 1.0) * 255.0
        )

        if self.args.training:
            f_reconstructions_ntc = self.decoder.forward_inter(
                ref_features,
                mc_hat[:, :64, :, :],
                mc_hat[:, 64:, :, :],
                context_qr_matrix_A,
                context_qr_matrix_NA,
                mv_qr_matrix_A,
                mv_qr_matrix_NA,
            )
            recon_image_ntc = self.frame_reconstruct(f_reconstructions_ntc)
            loss_ntc = self.squared_difference(
                input_image * 255.0, recon_image_ntc.clamp(0.0, 1.0) * 255.0
            )
            loss_ntc = loss_ntc.mean()
        else:
            loss_ntc = loss_G

        return (
            recon_image,
            CBR,
            chan_param,
            loss_G.mean(),
            loss_ntc.mean(),
            input_image,
            bpp,
        )

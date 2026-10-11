import math

import torch
import torch.nn as nn

from net.fvc_en_vlc import (
    DCN,
    ResnetBlock,
    Snr_attention_module,
    deform_groups,
    deform_ks,
    out_channel_F,
    out_channel_M,
    out_channel_O,
)
from net.video_net import UNet, flow_warp
from utils_entropy.ckbd import ckbd_merge, ckbd_split

dim1 = 64 * 3
dim2 = 64
dim3 = 128
g_ch_1x = dim2


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


class Channel_frame_decoder(nn.Module):
    def __init__(
        self,
        out_channel_M=dim3,
        rate_choice=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 32, 48, 64],
        C_channel=12,
    ):
        super(Channel_frame_decoder, self).__init__()
        # Resnet 抽取
        self.res1 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.res2 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.snr_attention1 = Snr_attention_module(out_channel_M)
        self.snr_attention2 = Snr_attention_module(out_channel_M)
        self.snr_attention3 = Snr_attention_module(C_channel)
        self.ra_decoder = RateAdaptionDecoder(C_channel, rate_choice)
        self.rate_num = len(rate_choice)
        self.rate_token = nn.Parameter(torch.zeros(self.rate_num, C_channel))

    def MIMO_layerdemapping(self, context, arranged_order):
        channels, order_length = arranged_order.shape
        rearranged_context = torch.empty(context.shape).cuda()
        rearranged_order = torch.empty(arranged_order.shape).int().cuda()
        for i in range(channels):
            for j in range(order_length):
                rearranged_order[i, arranged_order[i, j]] = j
            for k in range(8):
                rearranged_context[:, k * 4 + i * 32 : (k + 1) * 4 + i * 32, :, :] = (
                    context[
                        :,
                        rearranged_order[i, k] * 4 + i * 32 : (
                            rearranged_order[i, k] + 1
                        )
                        * 4
                        + i * 32,
                        :,
                        :,
                    ]
                )
        return rearranged_context

    def forward(
        self,
        input_feature,
        clip_map,
        snr,
        indexes,
        context_qr_matrix_A,
        context_qr_matrix_NA,
    ):
        B, _, H, W = input_feature.shape
        z = self.ra_decoder(input_feature, indexes)

        z_BLC = z.flatten(2).permute(0, 2, 1)
        rate_token = torch.index_select(self.rate_token, 0, indexes)  # BL, N
        rate_token = rate_token.reshape(B, H * W, z.shape[1])
        z_BLC = z_BLC + rate_token
        z = z_BLC.reshape(B, H, W, z.shape[1]).permute(0, 3, 1, 2)

        z = self.snr_attention3(
            z, clip_map, snr, context_qr_matrix_A, context_qr_matrix_NA
        )
        z = self.snr_attention1(
            self.res1(z), clip_map, snr, context_qr_matrix_A, context_qr_matrix_NA
        )
        z = self.snr_attention2(
            self.res2(z), clip_map, snr, context_qr_matrix_A, context_qr_matrix_NA
        )

        return z


class Channel_mv_decoder(nn.Module):
    def __init__(
        self,
        out_channel_M=dim3,
        rate_choice=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 32, 48, 64],
        C_channel=12,
    ):
        super(Channel_mv_decoder, self).__init__()
        # Resnet 抽取
        self.res1 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.res2 = ResnetBlock(out_channel_M, padding_type="zero", use_dropout=False)
        self.snr_attention1 = Snr_attention_module(out_channel_M)
        self.snr_attention2 = Snr_attention_module(out_channel_M)
        self.snr_attention3 = Snr_attention_module(C_channel)
        self.ra_decoder = RateAdaptionDecoder(C_channel, rate_choice)
        self.rate_num = len(rate_choice)
        self.rate_token = nn.Parameter(torch.zeros(self.rate_num, C_channel))

    def forward(
        self, input_feature, clip_map, snr, indexes, mv_qr_matrix_A, mv_qr_matrix_NA
    ):
        B, _, H, W = input_feature.shape
        z = self.ra_decoder(input_feature, indexes)

        z_BLC = z.flatten(2).permute(0, 2, 1)
        rate_token = torch.index_select(self.rate_token, 0, indexes)  # BL, N
        rate_token = rate_token.reshape(B, H * W, z.shape[1])
        z_BLC = z_BLC + rate_token
        z = z_BLC.reshape(B, H, W, z.shape[1]).permute(0, 3, 1, 2)

        z = self.snr_attention3(z, clip_map, snr, mv_qr_matrix_A, mv_qr_matrix_NA)
        z = self.snr_attention1(
            self.res1(z), clip_map, snr, mv_qr_matrix_A, mv_qr_matrix_NA
        )
        z = self.snr_attention2(
            self.res2(z), clip_map, snr, mv_qr_matrix_A, mv_qr_matrix_NA
        )

        return z


class ReconGeneration(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.first_conv = nn.Conv2d(dim1, dim2, 3, stride=1, padding=1)
        self.unet_1 = UNet(dim2, dim2, inplace=inplace)
        self.unet_2 = UNet(dim2, dim2, inplace=inplace)

    def forward(self, f_hat, context, mv):
        feature = self.first_conv(torch.cat((f_hat, context, mv), dim=1))
        feature = self.unet_1(feature)
        feature = self.unet_2(feature)
        return feature


class FVC_DECODER(nn.Module):
    def __init__(self):
        super().__init__()

        self.motion_compensation = nn.Sequential(
            nn.Conv2d(dim3, dim2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim2, dim2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim2, dim2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim2, dim2, kernel_size=3, stride=1, padding=1),
        )

        self.offset_mask_conv = nn.Conv2d(
            out_channel_O,
            deform_groups * 3 * deform_ks * deform_ks,
            kernel_size=deform_ks,
            stride=1,
            padding=1,
        )

        self.feature_refinement = ReconGeneration()

        self.deform_conv = DCN(
            out_channel_F,
            out_channel_F,
            kernel_size=(deform_ks, deform_ks),
            padding=deform_ks // 2,
            deform_groups=deform_groups,
        )

        self.context_decoder = Channel_frame_decoder(out_channel_M=dim2, C_channel=64)
        self.mv_decoder = Channel_mv_decoder(out_channel_M=dim2, C_channel=64)

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

    def forward(
        self,
        fr,
        mv_res,
        index_offset,
        index_context,
        clip_map,
        h,
        chan_param,
        context_qr_matrix_A,
        context_qr_matrix_NA,
        mv_qr_matrix_A,
        mv_qr_matrix_NA,
    ):
        mv = self.mv_decoder(
            mv_res[:, :64, :, :],
            clip_map,
            chan_param,
            index_offset,
            mv_qr_matrix_A,
            mv_qr_matrix_NA,
        )
        f_cur = self.context_decoder(
            mv_res[:, 64:, :, :],
            clip_map,
            chan_param,
            index_context,
            context_qr_matrix_A,
            context_qr_matrix_NA,
        )
        xr_cur = self.forward_inter(
            fr,
            mv,
            f_cur,
            context_qr_matrix_A,
            context_qr_matrix_NA,
            mv_qr_matrix_A,
            mv_qr_matrix_NA,
        )  # inter_frame f_base.view(64, 96//3*5)

        return xr_cur

    def forward_inter(
        self,
        fr,
        mv,
        f_cur,
        context_qr_matrix_A,
        context_qr_matrix_NA,
        mv_qr_matrix_A,
        mv_qr_matrix_NA,
    ):
        mv_A, mv_NA = ckbd_split(mv)
        mv_A = mv_A * mv_qr_matrix_A
        mv_NA = mv_NA * mv_qr_matrix_NA
        mv = ckbd_merge(mv_A, mv_NA)
        f_cur_A, f_cur_NA = ckbd_split(f_cur)
        f_cur_A = f_cur_A * context_qr_matrix_A
        f_cur_NA = f_cur_NA * context_qr_matrix_NA
        f_cur = ckbd_merge(f_cur_A, f_cur_NA)
        # motion compensation
        deformed_f_ref = self.dcn_warp(mv, fr)
        f_mc = torch.cat((deformed_f_ref, fr), dim=1)
        f_mc = self.motion_compensation(f_mc)
        f_predict = deformed_f_ref + f_mc
        f_cat = torch.cat((f_cur, f_predict, mv), dim=1)
        context = self.context_generation(fr, f_cat)
        # frame refinement
        f_combine = self.feature_refinement(f_cur, context, mv)
        return f_combine

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

    def dcn_warp(self, offset_info, f_ref):
        offset_and_mask = self.offset_mask_conv(offset_info)
        o1, o2, mask = torch.chunk(offset_and_mask, 3, dim=1)
        offset_map = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask.float())
        deformed_f_ref = self.deform_conv(f_ref.float(), offset_map.float(), mask)
        return deformed_f_ref

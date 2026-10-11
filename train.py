import argparse
import math
import os
import time
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

import lpips
import numpy as np
import pytorch_msssim
import scipy.io as scio
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

from data.datasets import UVGDataSet, Vimeo_data, get_testloader, get_trainloader
from loss.distortion import MS_SSIM
from net.network_img_vlc import Img_semantic
from net.network_vlc import WITT
from utils import AverageMeter, logger_configuration, seed_torch

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="WITT")
parser.add_argument("--training", default="True", help="training or testing")
parser.add_argument(
    "--distortion-metric",
    type=str,
    default="MSE",
    choices=["MSE", "MS-SSIM"],
    help="evaluation metrics",
)
parser.add_argument(
    "--model",
    type=str,
    default="WITT",
    choices=["WITT", "WITT_W/O"],
    help="WITT model or WITT without channel ModNet",
)
parser.add_argument(
    "--channel-type",
    type=str,
    default="MIMO",
    choices=["awgn", "rayleigh", "MIMO"],
    help="wireless channel model, awgn or rayleigh",
)
parser.add_argument(
    "--multiple-snr", type=str, default="0,2,4,6,8,10,12,14", help="random or fixed snr"
)
parser.add_argument(
    "--multiple-rate",
    type=str,
    default="2",  # '2, 4, 6, 8, 10', # 0.015:0.01:0.065
    help="random or fixed rate",
)
args = parser.parse_args()
scaler = GradScaler()


class config:
    local_rank = int(os.environ["LOCAL_RANK"])
    seed = 1024
    pass_channel = True
    CUDA = True
    multi_gpu = True
    norm = False
    # logger
    print_step = 2523
    filename = datetime.now().__str__()[:-9]
    workdir = "./history/{}".format(filename)
    log = workdir + "/Log_{}.log".format(filename)
    # samples = workdir + '/samples'
    # models = workdir + '/models'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = 600
    lamda = [0.015, 0.06, 0.12, 0.20, 0.32]
    lamda_num = [0, 1, 2, 3, 4]

    args.training = False

    # 设置当前程序使用的GPU。根据python的机制，在单卡训练时本质上程序只使用一个CPU核心，而DataParallel
    # 不管用了几张GPU，依然只用一个CPU核心，在分布式训练中，每一张卡分别对应一个CPU核心，即几个GPU几个CPU核心
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # 分布式初始化
    args.dist_url = "env://"  # 设置url
    args.dist_backend = "nccl"  # 通信后端，nvidia GPU推荐使用NCCL

    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
    dist.barrier()  # 等待所有进程都初始化完毕，即所有GPU都要运行到这一步以后在继续

    save_model_freq = 20
    train_data_dir = "GC/datasets/train_image_part"
    test_data_dir = "GC/datasets/test_image_part"

    batch_size = 16
    downsample = 4


CalcuSSIM = MS_SSIM(data_range=1.0, levels=4, channel=3).to(config.device)


def load_weights(model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=True)
    del pretrained


def interpolate_log(min_val, max_val, num, decending=True):
    assert max_val > min_val
    assert min_val > 0
    if decending:
        values = np.linspace(math.log(max_val), math.log(min_val), num)
    else:
        values = np.linspace(math.log(min_val), math.log(max_val), num)
    values = np.exp(values)
    return values


def train_one_epoch_img(args, H, epoch):
    net.train()
    train_loader = get_trainloader(args, train_dataset, config, epoch)
    elapsed, losses, psnrs, cbrs, snrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, losses, psnrs, cbrs, snrs]
    global global_step

    for batch_idx, (input, _) in enumerate(train_loader):
        start_time = time.time()
        global_step = global_step + 1
        # print(input.shape, ref.shape)
        input = input.to(config.device)

        with autocast():
            recon_image, CBR, SNR, mse, loss_G = net(input, epoch)
            loss = loss_G
            psnr = 10 * (torch.log(255.0 * 255.0 / mse) / np.log(10))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        elapsed.update(time.time() - start_time)
        losses.update(loss.item())
        cbrs.update(CBR)
        snrs.update(SNR)

        psnrs.update(psnr.item())

        if (global_step % config.print_step) == 0:
            if config.local_rank == 0:
                process = (
                    (global_step % train_loader.__len__())
                    / (train_loader.__len__())
                    * 100.0
                )
                log = " | ".join(
                    [
                        f"Epoch {epoch}",
                        f"Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]",
                        f"Time {elapsed.val:.3f}",
                        f"Loss {losses.val:.3f} ({losses.avg:.3f})",
                        f"CBR {cbrs.val:.4f} ({cbrs.avg:.4f})",
                        f"SNR {snrs.val:.1f} ({snrs.avg:.1f})",
                        f"PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})",
                    ]
                )
                logger.info(log)
                for i in metrics:
                    i.clear()
    for i in metrics:
        i.clear()


def train_one_epoch(args, H, epoch):
    net.train()
    train_loader = get_trainloader(args, train_dataset, config, epoch)
    elapsed, losses, losses_ntc, psnrs, cbrs, lamdas, snrs = [
        AverageMeter() for _ in range(7)
    ]
    metrics = [elapsed, losses, losses_ntc, psnrs, cbrs, snrs]
    global global_step

    for batch_idx, (input, ref) in enumerate(train_loader):
        start_time = time.time()
        global_step = global_step + 1
        # print(input.shape, ref.shape)
        input = input.to(config.device)
        ref = ref.to(config.device)
        with autocast():
            num = 0  # choice(config.lamda_num)
            mv_q_scales = net.module.encoder.mv_qr_scale[num]
            context_q_scales = net.module.encoder.context_qr_scale[num]
            # mv_q_scales = torch.clamp(mv_q_scales, 1, 100)
            # context_q_scales = torch.clamp(context_q_scales, 1, 100)
            # print(net.module.encoder.mv_qr_scale.max().item(), net.module.encoder.mv_qr_scale.min().item())
            # print(mv_q_scales.item(), context_q_scales.item())
            recon_image, CBR, SNR, loss_G, loss_ntc, input_im, bpp = net(
                input, ref, epoch, mv_q_scales, context_q_scales, 12
            )
            loss = config.lamda[num] * (loss_G + loss_ntc) + bpp
            psnr = 10 * (torch.log(255.0 * 255.0 / loss_G) / np.log(10))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # msssim = 1 - CalcuSSIM(input_im, recon_image.clamp(0., 1.)).mean().item()
        # lpips_value = lpips_loss(input_im, recon_image.clamp(0., 1.)).mean().item()

        optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        elapsed.update(time.time() - start_time)
        losses.update(loss_G.item())
        losses_ntc.update(loss_ntc.item())
        lamdas.update(config.lamda[num])
        cbrs.update(CBR)
        snrs.update(SNR)
        psnrs.update(psnr.item())
        # lpipss.update(lpips_value)
        # msssims.update(msssim)

        if (global_step % config.print_step) == 0:
            if config.local_rank == 0:
                process = (
                    (global_step % train_loader.__len__())
                    / (train_loader.__len__())
                    * 100.0
                )
                log = " | ".join(
                    [
                        f"Epoch {epoch}",
                        f"Lamda {lamdas.val:.4f} ({lamdas.avg:.4f})",
                        f"Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]",
                        f"Time {elapsed.val:.3f}",
                        f"Loss {losses.val:.3f} ({losses.avg:.3f})",
                        f"Loss_ntc {losses_ntc.val:.3f} ({losses_ntc.avg:.3f})",
                        f"CBR {cbrs.val:.4f} ({cbrs.avg:.4f})",
                        f"SNR {snrs.val:.1f} ({snrs.avg:.1f})",
                        f"PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})",
                        # f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        # f'LPIPS {lpipss.val:.3f} ({lpipss.avg:.3f})',
                    ]
                )
                logger.info(log)
                for i in metrics:
                    i.clear()
    for i in metrics:
        i.clear()


def test(H, test_dataset, epoch):
    config.isTrain = False
    net.eval()
    net_img.eval()
    elapsed, psnrs, msssims, lpipss, snrs, cbrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, psnrs, msssims, lpipss, snrs, cbrs]

    mv_q_scales = net.module.encoder.mv_qr_scale[0]
    context_q_scales = net.module.encoder.context_qr_scale[0]

    results_snr = np.zeros(len(mv_q_scales))
    results_cbr = np.zeros(len(mv_q_scales))
    results_psnr = np.zeros(len(mv_q_scales))
    results_msssim = np.zeros(len(mv_q_scales))
    results_lpips = np.zeros(len(mv_q_scales))

    with torch.no_grad():
        for i in range(len(mv_q_scales)):
            for batch_idx, (input, ref) in enumerate(test_loader):
                start_time = time.time()
                input = input.to(config.device)
                ref = ref.to(config.device)
                with autocast():
                    recon_ref, CBR, SNR, mse, loss_G, loss_ntc, bpp = net_img(
                        ref, epoch, 10
                    )  # recon_ref, CBR, SNR, mse, loss_G = net_img(ref, epoch, 12) # I frame

                    psnr = 10 * (torch.log(255.0 * 255.0 / mse) / np.log(10))
                    msssim = (
                        pytorch_msssim.ms_ssim(
                            ref,
                            recon_ref.clamp(0.0, 1.0),
                            data_range=1.0,
                            size_average=True,
                        )
                        .mean()
                        .item()
                    )  # 1 - CalcuSSIM(input_im, recon_image.clamp(0., 1.)).mean().item()
                    lpips_value = (
                        lpips_loss(ref, recon_ref.clamp(0.0, 1.0)).mean().item()
                    )

                lpipss.update(lpips_value)
                elapsed.update(time.time() - start_time)
                cbrs.update(CBR)
                snrs.update(SNR)
                psnrs.update(psnr.item())
                msssims.update(msssim)
                ref = recon_ref.clamp(0.0, 1.0)
                # ref = ref.type(torch.float32)

                for j in range(input.shape[1] - 1):  # P frame
                    input_cur = input[:, j + 1, :, :, :]
                    with autocast():
                        recon_image, CBR, SNR, loss_G, loss_ntc, input_im, bpp = net(
                            input_cur,
                            ref,
                            epoch,
                            mv_q_scales[i],
                            context_q_scales[i],
                            10,
                        )
                        psnr = 10 * (torch.log(255.0 * 255.0 / loss_G) / np.log(10))
                        msssim = (
                            pytorch_msssim.ms_ssim(
                                input_im,
                                recon_image.clamp(0.0, 1.0),
                                data_range=1.0,
                                size_average=True,
                            )
                            .mean()
                            .item()
                        )  # 1 - CalcuSSIM(input_im, recon_image.clamp(0., 1.)).mean().item()
                        lpips_value = (
                            lpips_loss(input_im, recon_image.clamp(0.0, 1.0))
                            .mean()
                            .item()
                        )
                    """
                    if batch_idx == 15 and j == 8 and i == 1 and config.local_rank == 0: #
                        print(psnr, msssim, CBR)
                        save_image(recon_image,'./JSAC/img_D_cvst.png')
                    """
                    lpipss.update(lpips_value)
                    ref = recon_image.clamp(0.0, 1.0)

                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)

                    psnrs.update(psnr.item())
                    msssims.update(msssim)
                    """
                    if config.local_rank == 0:
                        log = (' | '.join([
                            f'Time {elapsed.val:.3f}',
                            f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                            f'SNR {snrs.val:.1f}',
                            f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                            f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                            f'LPIPS {lpipss.val:.3f} ({lpipss.avg:.3f})',
                        ]))
                    if config.local_rank == 0:
                        logger.info(log)
                    """

            results_snr[i] = snrs.avg
            results_cbr[i] = cbrs.avg
            results_psnr[i] = psnrs.avg
            results_msssim[i] = msssims.avg
            results_lpips[i] = lpipss.avg

            for t in metrics:
                t.clear()
    """
    if config.local_rank == 0:
        print("Test Results:")
        print("SNR: {}" .format(results_snr.tolist()))
        print("CBR: {}".format(results_cbr.tolist()))
        print("PSNR: {}" .format(results_psnr.tolist()))
        print("MS-SSIM: {}".format(results_msssim.tolist()))
        print("LPIPS: {}".format(results_lpips.tolist()))
    """
    print(
        "CBR: {}".format(results_cbr.tolist()), "PSNR: {}".format(results_psnr.tolist())
    )
    # print("MS-SSIM: {}".format(results_msssim.tolist()))


if __name__ == "__main__":
    seed_torch()
    if config.local_rank == 0:
        logger = logger_configuration(config, save_log=True)
        logger.info(config.__dict__)

    f = "H_MIMO8.mat"
    Hf = scio.loadmat(f)
    # Hf = h5py.File(f)
    # Hf = mat73.loadmat(f)
    # print(Hf['Hnew'])
    Hf = np.complex64(Hf["Hnew"])
    Hf = torch.from_numpy(Hf)
    # index = torch.randperm(Hf.shape[2])
    # Hf = Hf[:,:, index]
    index = np.load("mimo_seq.npy")
    Hf = Hf[:, :, index]
    Hf_train = Hf[:, :, 0:5000]
    Hf_test = Hf[:, :, 5000 : Hf.shape[2]]

    net_img = Img_semantic(args, config)
    net = WITT(args, config)
    # net = Video_semantic(args, config)

    lpips_loss = lpips.LPIPS(net="vgg", verbose=False).cuda()
    lpips_loss.eval()

    model_save = "./cvst_0.015.pth"
    model_path_base = "./cvst_0.015.pth"  # context1
    model_path = "./base_img_mimo_vlc.pth"
    """
    input1 = torch.randn(1,4,3,128,128).cuda()
    flops, params = profile(net.to(config.device), inputs=(input1, 0))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
    """

    codec = torch.load(model_path_base, map_location="cuda")
    model_dict = net.state_dict()
    state_dict = {
        k: v
        for k, v in codec.items()
        if k in model_dict.keys() and model_dict[k].shape == v.shape
    }
    model_dict.update(state_dict)
    net.load_state_dict(model_dict)

    codec = torch.load(model_path, map_location="cuda")  # autocast
    model_dict = net_img.state_dict()
    state_dict = {k: v for k, v in codec.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    net_img.load_state_dict(model_dict)

    if config.multi_gpu:
        net = net.to(config.device)
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[config.local_rank],
            output_device=config.local_rank,
            broadcast_buffers=True,
            find_unused_parameters=True,
        )
    else:
        net = net.to(config.device)

    if config.multi_gpu:
        net_img = net_img.to(config.device)
        net_img = torch.nn.parallel.DistributedDataParallel(
            net_img,
            device_ids=[config.local_rank],
            output_device=config.local_rank,
            broadcast_buffers=True,
            find_unused_parameters=True,
        )
    else:
        net_img = net_img.to(config.device)

    # net.module.load_state_dict(torch.load(model_path))

    global_step = 0
    # steps_epoch = global_step // train_loader.__len__()
    steps_epoch = 0

    train_dataset = Vimeo_data()
    test_dataset = UVGDataSet(
        root="HEVC/ClassC_images/", filelist="HEVC/ClassC/originalv.txt"
    )
    test_loader = get_testloader(args, test_dataset, config, 0)

    if args.training:
        for epoch in range(steps_epoch, config.tot_epoch):
            if epoch <= 100:
                optimizer = torch.optim.AdamW(
                    net.parameters(), lr=0.00002, weight_decay=0.0
                )
            elif epoch <= 600:
                optimizer = torch.optim.AdamW(
                    net.parameters(), lr=0.00002, weight_decay=0.0
                )
            else:
                optimizer = torch.optim.AdamW(
                    net.parameters(), lr=0.00002, weight_decay=0.0
                )

            """
            if epoch <= 300:
                for param in net.module.parameters():
                    param.requires_grad = False
                for param in net.module.encoder.context_encoder.parameters():
                    param.requires_grad = True
                for param in net.module.decoder.context_decoder.parameters():
                    param.requires_grad = True

                for param in net.module.encoder.clip.parameters():
                    param.requires_grad = True
                for param in net.module.decoder.fusion.parameters():
                    param.requires_grad = True
                for param in net.module.decoder.feature_refinement.parameters():
                    param.requires_grad = True
                for param in net.module.decoder.conv_offset.parameters():
                    param.requires_grad = True
                for param in net.module.frame_reconstruct.parameters():
                    param.requires_grad = True

                for param in net.module.encoder.mv_encoder.parameters():
                    param.requires_grad = True
                for param in net.module.decoder.mv_decoder.parameters():
                    param.requires_grad = True
                for param in net.module.decoder.motion_compensation.parameters():
                    param.requires_grad = True
                for param in net.module.decoder.offset_mask_conv.parameters():
                    param.requires_grad = True
                for param in net.module.decoder.deform_conv.parameters():
                    param.requires_grad = True
            """
            if (epoch + 1) % config.save_model_freq == 0 or (epoch + 1) % 3 == 0:
                torch.save(net.module.state_dict(), model_save)
                # test(Hf_test, test_dataset, 0)
            # if (epoch + 1) % 10 == 0 and (epoch + 1) <=100:
            #     torch.save(net.module.state_dict(), model_save)

            # train_one_epoch_img(args, Hf_train, epoch)
            train_one_epoch(args, Hf_train, epoch)
            if (epoch + 1) % config.save_model_freq == 0:
                test(Hf_test, test_dataset, 0)

            torch.cuda.empty_cache()  # 释放显存
    else:
        test(Hf_test, test_dataset, 0)

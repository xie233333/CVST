from torch.utils.data import Dataset
from PIL import Image
import cv2
import os
import numpy as np
from glob import glob
import torchvision
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset, TensorDataset
from data.video import VideoFolder
import torch
import math
import torch.utils.data as data
from data.augmentation import random_flip, random_crop_and_pad_image_and_labels
import imageio
import random
import torch.nn.functional as F


def random_crop_and_pad_image_and_labels_vimeo(image, labels, size):
    combined = torch.cat([image, labels], 0)
    last_image_dim = image.size()[0]
    image_shape = image.size()
    combined_pad = F.pad(
        combined,
        (
            0,
            max(size[1], image_shape[2]) - image_shape[2],
            0,
            max(size[0], image_shape[1]) - image_shape[1],
        ),
    )
    freesize0 = random.randint(0, max(size[0], image_shape[1]) - size[0])
    freesize1 = random.randint(0, max(size[1], image_shape[2]) - size[1])
    combined_crop = combined_pad[
        :, freesize0 : freesize0 + size[0], freesize1 : freesize1 + size[1]
    ]
    return combined_crop[:last_image_dim, :, :], combined_crop[last_image_dim:, :, :]


def random_flip_vimeo(images, labels):
    # augmentation setting....
    horizontal_flip = 1
    vertical_flip = 1
    transforms = 1

    if transforms and vertical_flip and random.randint(0, 1) == 1:
        # 对tensor进行翻转，torch.flip(input,dim):第一个参数是输入，第二个参数是输入的第几维度，按照维度对输入进行翻转
        images = torch.flip(images, [1])
        labels = torch.flip(labels, [1])
    if transforms and horizontal_flip and random.randint(0, 1) == 1:
        images = torch.flip(images, [2])
        labels = torch.flip(labels, [2])

    return images, labels


def random_crop_and_pad_image_vimeo_gops(image, size):
    combined = image
    image_shape = image.size()
    combined_pad = F.pad(
        combined,
        (
            0,
            max(size[1], image_shape[3]) - image_shape[3],
            0,
            max(size[0], image_shape[2]) - image_shape[2],
        ),
    )
    freesize0 = random.randint(0, max(size[0], image_shape[2]) - size[0])
    freesize1 = random.randint(0, max(size[1], image_shape[3]) - size[1])
    combined_crop = combined_pad[
        :, :, freesize0 : freesize0 + size[0], freesize1 : freesize1 + size[1]
    ]
    return combined_crop


def random_flip_vimeo_gops(images):
    # augmentation setting....
    horizontal_flip = 1
    vertical_flip = 1
    transforms = 1

    if transforms and vertical_flip and random.randint(0, 1) == 1:
        # 对tensor进行翻转，torch.flip(input,dim):第一个参数是输入，第二个参数是输入的第几维度，按照维度对输入进行翻转
        images = torch.flip(images, [1])
    if transforms and horizontal_flip and random.randint(0, 1) == 1:
        images = torch.flip(images, [2])
    return images


class UVGDataSet(data.Dataset):
    def __init__(self, root="UVG/images/", filelist="UVG/originalv.txt", testfull=True):
        with open(filelist) as f:
            folders = f.readlines()
        self.ref = []
        self.input = []
        gop_size = 12
        for folder in folders:
            seq = folder.rstrip()
            imlist = os.listdir(os.path.join(root, seq))
            cnt = 0
            for im in imlist:
                if im[-4:] == ".png":
                    cnt += 1

            if testfull:
                framerange = cnt // gop_size
                if cnt % gop_size > 0:
                    framerange += 1
                gop = gop_size
            else:
                framerange = 1
                gop = gop_size

            for i in range(framerange):
                refpath = os.path.join(
                    root, seq, "im" + str(i * gop + 1).zfill(3) + ".png"
                )
                inputpath = []
                for j in range(gop):
                    if os.path.exists(
                        os.path.join(
                            root, seq, "im" + str(i * gop + j + 1).zfill(3) + ".png"
                        )
                    ):
                        inputpath.append(
                            os.path.join(
                                root, seq, "im" + str(i * gop + j + 1).zfill(3) + ".png"
                            )
                        )
                    else:
                        break
                self.ref.append(refpath)
                self.input.append(inputpath)

    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        ref_image = (
            imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32)
            / 255.0
        )
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        refout = dict()
        for filename in self.input[index]:
            input_image = (
                (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(
                    np.float32
                )
                / 255.0
            )
            input_images.append(input_image)

        input_images = np.array(input_images)
        # print(input_images.shape, ref_image.shape)
        return input_images, ref_image


"""
class Vimeo_data(data.Dataset):
    def __init__(self, path="vimeo_90k/sep_trainlist.txt", im_height=256, im_width=256):
        self.image_input_list = self.get_vimeo(filefolderlist=path)
        self.image_input_list.sort()
        self.im_height = im_height
        self.im_width = im_width


    def get_vimeo(self, rootdir="vimeo_90k/sequences/", filefolderlist="vimeo_90k/sep_trainlist.txt"):
        fns_train_input = []
        with open(filefolderlist) as f:
            data = f.readlines()

        for n, line in enumerate(data, 1):
            y = os.path.join(rootdir, line.rstrip())
            fns_train_input += [y]
        # print(fns_train_input)
        return fns_train_input

    def __len__(self):
        return len(self.image_input_list)

    def __getitem__(self, index):
        input_image = torch.zeros(7, 3, 256, 448)
        for i in range(1, 8):
            input_im = imageio.imread(os.path.join(self.image_input_list[index], 'im' + i.__str__() + '.png'))
            input_im = input_im.astype(np.float32) / 255.0
            input_im = input_im.transpose(2, 0, 1)
            input_im = torch.from_numpy(input_im).float()
            input_image[i - 1] = input_im
        input_image = random_crop_and_pad_image_vimeo_gops(input_image, [self.im_height, self.im_width])
        input_image = random_flip_vimeo_gops(input_image)
        return input_image  # GOP_size Channel H W
"""


class Vimeo_data(data.Dataset):
    def __init__(self, path="vimeo_90k/test.txt", im_height=256, im_width=256):
        self.image_input_list, self.image_ref_list = self.get_vimeo(filefolderlist=path)
        self.im_height = im_height
        self.im_width = im_width

        print("dataset find image: ", len(self.image_input_list))

    def get_vimeo(
        self, rootdir="vimeo_90k/sequences/", filefolderlist="vimeo_90k/test.txt"
    ):
        with open(filefolderlist) as f:
            data = f.readlines()

        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(data, 1):
            y = os.path.join(rootdir, line.rstrip())
            fns_train_input += [y]
            refnumber = int(y[-5:-4]) - 2
            refname = y[0:-5] + str(refnumber) + ".png"
            fns_train_ref += [refname]
        return fns_train_input, fns_train_ref

    def __len__(self):
        return len(self.image_input_list)

    def __getitem__(self, index):
        input_image = imageio.imread(self.image_input_list[index])
        ref_image = imageio.imread(self.image_ref_list[index])

        input_image = input_image.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0

        input_image = input_image.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)

        input_image = torch.from_numpy(input_image).float()
        ref_image = torch.from_numpy(ref_image).float()

        input_image, ref_image = random_crop_and_pad_image_and_labels_vimeo(
            input_image, ref_image, [self.im_height, self.im_width]
        )
        # 上述random_crop_and_pad_image_and_labels生成的input_image, ref_image 均为(3,256,256)
        input_image, ref_image = random_flip_vimeo(input_image, ref_image)
        # print(input_image.shape, ref_image.shape)
        return input_image, ref_image


class Datasets(Dataset):
    def __init__(self, data_dir, train):
        self.data_dir = data_dir
        self.train = train
        self.image_input_list = self.get_multi_vimeo(rootdir=self.data_dir)

    def __len__(self):
        return len(self.image_input_list)

    def __getitem__(self, index):
        input_images = self.image_input_list[index]
        return input_images

    def get_multi_vimeo(self, rootdir):
        # 遍历输入文件夹中的所有视频文件
        fns_train_input = []
        for filename in os.listdir(rootdir):
            if filename.endswith(".mp4") or filename.endswith(".avi"):
                video_path = os.path.join(rootdir, filename)

                # 打开视频文件
                video = cv2.VideoCapture(video_path)

                # 逐帧读取视频并保存为图像文件
                frame_count = 0
                input_list = []
                while True:
                    # 读取一帧
                    ret, frame = video.read()
                    # 如果没有帧可读，跳出循环
                    if not ret:
                        break

                    # 保存为图像文件
                    if frame_count % 3 == 0:
                        input_list.append(frame)
                    if frame_count % 21 == 18:
                        input_list = np.array(input_list).astype(np.float32) / 255.0
                        input_list = input_list.transpose(0, 3, 1, 2)
                        input_list = torch.Tensor(input_list).float()
                        if self.train:
                            input_list = random_crop_and_pad_image_and_labels(
                                input_list[0].unsqueeze(0), input_list, [128, 128]
                            )
                            input_list = random_flip(input_list)
                        fns_train_input.append(input_list)
                        input_list = []
                    # 增加帧计数
                    frame_count += 1

                # 释放视频文件
                video.release()

        return fns_train_input


def get_trainloader(args, train_dataset, config, epoch):
    # train_dataset = Datasets(data_dir = 'TCSVT/datasets/UCF101_train', train = True)
    # test_dataset = Datasets(data_dir = 'TCSVT/datasets/UCF101_test', train = False)
    # train_dataset = Datasets(data_dir = 'GC/datasets/train_gc', train = True)
    # test_dataset = Datasets(data_dir = 'GC/datasets/test_gc', train = False)
    # train_dataset = Vimeo_data()
    # train_dataset = UVGDataSet()

    # 2. DistributedSampler
    # 给每个rank对应的进程分配训练的样本索引，比如一共800样本8张卡，那么每张卡对应分配100个样本
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True
    )
    train_sampler.set_epoch(epoch)
    # 3. BatchSampler
    # 刚才每张卡分了100个样本，假设BatchSize=16，那么能分成100/16=6...4，即多出4个样本
    # 下面的drop_last=True表示舍弃这四个样本，False将剩余4个样本为一组（注意最后就不是6个一组了）
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, config.batch_size, drop_last=True
    )

    # 4. DataLoader
    # 验证集没有采用batchsampler,因此在dataloader中使用batch_size参数即可
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, pin_memory=True, num_workers=8
    )

    return train_loader


def get_testloader(args, test_dataset, config, epoch):
    # train_dataset = Datasets(data_dir = 'TCSVT/datasets/UCF101_train', train = True)
    # test_dataset = Datasets(data_dir = 'TCSVT/datasets/UCF101_test', train = False)
    # train_dataset = Datasets(data_dir = 'GC/datasets/train_gc', train = True)
    # test_dataset = Datasets(data_dir = 'GC/datasets/test_gc', train = False)
    # train_dataset = Vimeo_data()
    # train_dataset = UVGDataSet()

    # 2. DistributedSampler
    # 给每个rank对应的进程分配训练的样本索引，比如一共800样本8张卡，那么每张卡对应分配100个样本

    # 3. BatchSampler
    # 刚才每张卡分了100个样本，假设BatchSize=16，那么能分成100/16=6...4，即多出4个样本
    # 下面的drop_last=True表示舍弃这四个样本，False将剩余4个样本为一组（注意最后就不是6个一组了）
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, shuffle=False
    )

    # 4. DataLoader
    # 验证集没有采用batchsampler,因此在dataloader中使用batch_size参数即可
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        sampler=test_sampler,
        drop_last=True,
        pin_memory=True,
        num_workers=8,
    )

    return test_loader

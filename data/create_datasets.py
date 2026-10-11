import numpy as np
import torchvision
import torch
from PIL import Image
import os
import cv2

rootdir = "TCSVT/datasets/UCF101_test"
output_folder = "TCSVT/datasets/UCF_test_gop7/111"

# 如果目标文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_count = 0

for filename in os.listdir(rootdir):
    if filename.endswith(".mp4") or filename.endswith(".avi"):
        video_path = os.path.join(rootdir, filename)

        # 打开视频文件
        video = cv2.VideoCapture(video_path)
        # 逐帧读取视频并保存为图像文件

        input_list = []
        while True:
            # 读取一帧
            ret, frame = video.read()
            # 如果没有帧可读，跳出循环
            if not ret:
                break
            # 保存为图像文件
            if frame_count % 2 == 0:
                frame = Image.fromarray(frame)
                frame.save(
                    f"TCSVT/datasets/UCF_test_gop7/111/frame_{frame_count // 2}.png"
                )
                # 增加帧计数
            frame_count += 1

        # 释放视频文件
        video.release()

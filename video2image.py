import os
import cv2


def batch_video_to_images(input_folder, output_folder):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有视频文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            video_path = os.path.join(input_folder, filename)

            # 打开视频文件
            video = cv2.VideoCapture(video_path)

            # 确保视频文件成功打开
            if not video.isOpened():
                print(f"无法打开视频文件: {filename}")
                continue

            # 逐帧读取视频并保存为图像文件
            frame_count = 0
            while True:
                # 读取一帧
                ret, frame = video.read()

                # 如果没有帧可读，跳出循环
                if not ret:
                    break

                # 保存为图像文件
                if frame_count % 5 == 0:
                    frame_path = os.path.join(
                        output_folder, f"{filename}_{frame_count}.jpg"
                    )
                    cv2.imwrite(frame_path, frame)

                # 增加帧计数
                frame_count += 1

            # 释放视频文件
            video.release()


# 测试代码
input_folder = "GC/datasets/test_gc"
output_folder = "GC/datasets/test_gc_image/1"
batch_video_to_images(input_folder, output_folder)

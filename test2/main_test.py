# -*- coding: utf-8 -*-

import argparse
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import os
from skimage.exposure import rescale_intensity
from noise_gen import add_blocky_speckle_noise
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='models/DnCNN_oct', help='directory of the model')
    parser.add_argument('--model_name', default='final_model.pth', help='the model name')
    parser.add_argument('--test_dir', default='test_img', help='directory of test images')
    return parser.parse_args()


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def show_comparison(noisy, original, denoised):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(noisy, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(denoised, cmap='gray')
    plt.title('Denoised Image')
    plt.axis('off')

    plt.show()




# 定义 DnCNN 模型
class DnCNN(nn.Module):
    def __init__(
        self,
        Firstdepth=1,
        n_channels=64,
        image_channels=1,
        use_bnorm=True,
        kernel_size=3,
    ):
        super(DnCNN, self).__init__()

        layers1 = []
        layers2 = []
        layers3 = []
        layers4 = []
        layers5 = []

        layers1.append(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True,
            )
        )
        layers1.append(nn.ReLU(inplace=True))

        layers1.append(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False
            )
        )
        layers1.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers1.append(nn.ReLU(inplace=True))
        layers1.append(nn.MaxPool2d(2))

        layers2.append(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False
            )
        )
        layers2.append(nn.BatchNorm2d(128, eps=0.0001, momentum=0.90))
        layers2.append(nn.ReLU(inplace=True))
        layers2.append(
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False
            )
        )
        layers2.append(nn.BatchNorm2d(128, eps=0.0001, momentum=0.90))
        layers2.append(nn.ReLU(inplace=True))
        layers2.append(nn.MaxPool2d(2))

        layers3.append(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False
            )
        )
        layers3.append(nn.BatchNorm2d(256, eps=0.0001, momentum=0.90))
        layers3.append(nn.ReLU(inplace=True))

        layers4.append(
            nn.Conv2d(
                in_channels=256, out_channels=128, kernel_size=3, padding=1, bias=False
            )
        )
        layers4.append(nn.ReLU(inplace=True))
        layers4.append(
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False
            )
        )
        layers4.append(nn.ReLU(inplace=True))
        layers4.append(
            nn.Conv2d(
                in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False
            )
        )
        layers4.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers4.append(nn.ReLU(inplace=True))

        layers5.append(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False
            )
        )
        layers5.append(nn.ReLU(inplace=True))
        layers5.append(
            nn.Conv2d(
                in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=False
            )
        )

        self.FirstOut = nn.Sequential(*layers1)
        self.SecondOut = nn.Sequential(*layers2)
        self._3out = nn.Sequential(*layers3)
        self._4Out = nn.Sequential(*layers4)
        self._5Out = nn.Sequential(*layers5)

        self._initialize_weights()

    def forward(self, x):
        y = x
        x1 = self.FirstOut(x)
        x2 = self.SecondOut(x1)
        x3 = self._3out(x2)
        x4 = torch.nn.functional.interpolate(x3, size=(265, 480), mode="nearest")
        x5 = self._4Out(x4)
        x5 = x5 - x1
        x6 = torch.nn.functional.interpolate(x5, size=(530, 960), mode="nearest")
        x7 = self._5Out(x6)
        return y - x7

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(
                    m.weight, nonlinearity="relu"
                )  # 使用 Kaiming 初始化
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    args = parse_args()

    model = torch.load(os.path.join(args.model_dir, args.model_name))  # 加载状态字典
    log('load trained model')

    model.eval()  # evaluation mode
    if torch.cuda.is_available():
        model = model.cuda()

    for img_name in os.listdir(args.test_dir):
        if img_name.endswith('Raw Image.tif'):
            noisy_img_path = os.path.join(args.test_dir, img_name)
            original_img_path = noisy_img_path.replace('Raw Image.tif', 'Averaged Image.tif')

            noisy_image = cv2.imread(noisy_img_path)
            
            original_image = np.array(imread(original_img_path), dtype=np.float32)  

            # noisy_image = cv2.imread(original_img_path)
            # noisy_image =  add_blocky_speckle_noise(
            #     noisy_image, block_size=2, noise_intensity=0.1
            # )
            
            noisy_image=np.array(noisy_image, dtype=np.float32) 
            # 转换为灰度图
            if len(noisy_image.shape) == 3:  # 如果是彩色图像
                noisy_image = rgb2gray(noisy_image)
            if len(original_image.shape) == 3:  # 如果是彩色图像
                original_image = rgb2gray(original_image)

            # Prepare the noisy image for the model
            noisy_tensor = torch.from_numpy(noisy_image).unsqueeze(0).unsqueeze(0)
            if torch.cuda.is_available():
                noisy_tensor = noisy_tensor.cuda()

            # Denoising
            with torch.no_grad():
                denoised_tensor = model(noisy_tensor)

            denoised_image = denoised_tensor.squeeze().cpu().numpy()

            # Show comparison
            show_comparison(noisy_image, original_image, denoised_image)
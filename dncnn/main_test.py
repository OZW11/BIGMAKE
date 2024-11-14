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
    def __init__(self, Firstdepth=1,Seconddepth=4, n_channels=32, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers1 = []
        layers2 = []
        layers3 = []
        
        
        layers1.append(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers1.append(nn.ReLU(inplace=True))
        for _ in range(Firstdepth - 2):
            layers1.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers1.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.90))
            layers1.append(nn.ReLU(inplace=True))



        for _ in range(Seconddepth - 2):
            layers2.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers2.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.90))
            layers2.append(nn.ReLU(inplace=True))
        layers3.append(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        
        

        self.FirstOut = nn.Sequential(*layers1)
        self.SecondOut = nn.Sequential(*layers2)
        self.LastondOut = nn.Sequential(*layers3)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out1 = self.FirstOut(x)
        out2 = self.SecondOut(out1)
        out2 = out2+out1
        out3 = self.SecondOut(out2)+out2
        out = self.LastondOut(out3)
        return y - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # 使用 Kaiming 初始化
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

            noisy_image = np.array(imread(noisy_img_path), dtype=np.float32) / 255.0
            original_image = np.array(imread(original_img_path), dtype=np.float32) / 255.0

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
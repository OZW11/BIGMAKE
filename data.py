# 时间： 2024/11/14 19:48
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import random
from itertools import chain


class My_Model_Dataset(Dataset):
    def __init__(self, args, image_dir, noise_dir=None, mode='train'):
        """
        初始化数据集类
        :param args: 配置参数，包括噪声强度等
        :param image_dir: 原始图像所在目录
        :param noise_dir: 测试模式下的噪声图像目录（训练模式不需要）
        :param mode: 'train' 或 'test'，决定数据集的用途
        """
        self.args = args  # 配置参数
        self.mode = mode  # 模式
        self.patch_size = args.patch_size  # 图像块大小
        self.nose_level = args.sigma  # 噪声水平
        self.n_patches = args.n_pat_per_image  # 每张图像的小块数量

        # 加载图像文件列表
        if self.mode == 'train':
            self.data_list = self.generate_train_data(image_dir)
        elif self.mode == 'test':
            self.data_list, self.noisy_list, self.name_list = self.generate_test_data(image_dir, noise_dir)

    def __getitem__(self, index):
        """获取数据集中的一个样本，并将其转换为张量格式"""
        clean_img = torch.tensor(self.data_list[index], dtype=torch.float32).unsqueeze(0) / 255.0
        if self.mode == 'train':
            noisy_img = self.add_noise_to_patch(clean_img)
            return clean_img, noisy_img
        elif self.mode == 'test':
            noisy_img = torch.tensor(self.noisy_list[index], dtype=torch.float32).unsqueeze(0) / 255.0
            return clean_img, noisy_img, self.name_list[index]

    def __len__(self):
        """返回数据集的长度"""
        return len(self.data_list)

    def generate_train_data(self, image_dir):
        """加载训练图像并生成patch块"""
        img_list = []
        for img_name in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度图
            patches = self.gen_patches(img,
                                       patch_size=self.patch_size,
                                       n=self.n_patches)
            img_list.append(patches)

        # 将所有图像块展平成一个列表
        img_list = list(chain(*img_list))

        return img_list

    def generate_test_data(self, image_dir, noise_dir):
        """为测试生成图像并返回图像和文件名"""
        img_list = []
        noisy_list = []
        name_list = []

        for clean_name, noisy_name in zip(os.listdir(image_dir), os.listdir(noise_dir)):
            clean_path = os.path.join(image_dir, clean_name)
            noisy_path = os.path.join(noise_dir, noisy_name)

            clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
            noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)

            img_list.append(clean_img)
            noisy_list.append(noisy_img)
            name_list.append(os.path.basename(clean_path))

        return img_list, noisy_list, name_list

    def gen_patches(self, img, patch_size=64, n=64):
        """
        从单张图像生成多个图像块
        :param img: 输入图像
        :param patch_size: 图像块大小
        :param n: 每张图像生成的图像块数量
        :return: 图像块列表
        """
        patches = []
        ih, iw = img.shape

        for _ in range(n):
            iy = random.randrange(0, ih - patch_size + 1)  # 随机选择图像块的起始 y 坐标
            ix = random.randrange(0, iw - patch_size + 1)  # 随机选择图像块的起始 x 坐标
            patch = img[iy:iy + patch_size, ix:ix + patch_size]

            # 将patch添加到patches列表中
            patches.append(patch)
        return patches

    def add_noise_to_patch(self, patch):
        """为每个patch添加噪声"""
        # 将PyTorch张量转换为NumPy数组
        patch_np = patch.squeeze(0).numpy() * 255.0  # 反归一化到原始像素范围

        # 添加高斯噪声
        mean = 0
        sigma = self.nose_level
        gaussian_noise = np.random.normal(mean, sigma, patch_np.shape)
        noisy_patch = np.clip(patch_np + gaussian_noise, 0, 255)  # 保证像素值在0-255之间

        # 将NumPy数组转换回PyTorch张量
        noisy_patch_tensor = torch.tensor(noisy_patch, dtype=torch.float32).unsqueeze(0) / 255.0
        return noisy_patch_tensor


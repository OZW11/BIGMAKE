# -*- coding: utf-8 -*-
import argparse
import os
import time
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from noise_gen import add_blocky_speckle_noise
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 参数设置
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=70, type=int, help='batch size')
parser.add_argument('--train_data', default='data\Train400', type=str, help='path of train data')
parser.add_argument('--epoch', default=18000, type=int, help='number of train epochs')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate for Adam')
parser.add_argument('--model_dir', default='models/DnCNN_oct', help='directory of the model')
parser.add_argument('--model_name', default='final_model.pth', help='the model name')
parser.add_argument('--test_dir', default='test_img', help='directory of test images')
args = parser.parse_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()  # 检查是否有可用的 GPU
n_epoch = args.epoch
save_dir = os.path.join('models', args.model + '_oct')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

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

def random_cut(gray_image,target_size,x,y):
            # 获取图像的当前尺寸
            height, width = gray_image.shape


            crop_x, crop_y = x, y  # 这里的x和y应该是你希望裁剪的起始位置
            crop_width = min(width - crop_x, target_size[0])
            crop_height = min(height - crop_y, target_size[1])


            cropped_image = gray_image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

            if cropped_image.shape != target_size:
                resized_image = cv2.resize(cropped_image, target_size)
            else:
                resized_image = cropped_image  # 如果裁剪区域恰好是256x256，则直接使用它
            return resized_image

# 损失函数定义
class sum_squared_error(nn.Module):
    def __init__(self):
        super(sum_squared_error, self).__init__()

    def forward(self, input, target):
        return (torch.nn.functional.mse_loss(input, target, reduction='sum').div_(2))

# 加载图像函数，调整所有图像大小
def load_images(train_data_dir, target_size=(256, 256)):
    original_images = []
    noisy_images = []

    for filename in os.listdir(train_data_dir):
#        if 'Averaged' in filename:
        if 1:
            # 读取原始图像并转换为灰度图

            # 读取噪声图像并转换为灰度图
            image_path = os.path.join(train_data_dir, filename)
            image = cv2.imread(image_path)

            # 添加块状散斑噪声
            noisy_image = add_blocky_speckle_noise(image, block_size=5, noise_intensity=0.05)
            noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
            original_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


            # height, width = original_image.shape
            
            # randx=random.randint(0, width - 1-target_size[0])
            # randy=random.randint(0, height - 1-target_size[1])
            # original_image=random_cut(original_image,target_size,randx,randy)
            # noisy_image=random_cut(noisy_image,target_size,randx,randy)

            # cv2.imshow("check2",noisy_image)
            # cv2.imshow("check1",original_image)
            
            original_images.append(original_image)
            noisy_images.append(noisy_image)

    return np.array(noisy_images), np.array(original_images)

if __name__ == '__main__':
    # 模型构建

    print('===> 构建模型')
    #model = DnCNN(image_channels=1)  # 确保模型接受单通道图像
    model = torch.load(os.path.join(args.model_dir, args.model_name))  # 加载状态字典
    # 如果有初始训练轮次，则从指定轮次开始恢复模型
    initial_epoch = 0
    if initial_epoch > 0:
        print('通过加载 epoch %03d 恢复模型' % initial_epoch)
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
    model.train()

    # 定义损失函数
    criterion = sum_squared_error()

    # 将模型移到 GPU 上
    if cuda:
        model = model.cuda()

    # 定义优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)
    

    # 开始训练
    for epoch in range(initial_epoch, n_epoch):
        # 使用 DataLoader 进行批次训练
        start_time = time.time()
        
        noisy_images, original_images = load_images(args.train_data)
        # 增加通道维度，并将数据移到 GPU 上
        noisy_images = torch.from_numpy(np.expand_dims(noisy_images, axis=1)).float()
        original_images = torch.from_numpy(np.expand_dims(original_images, axis=1)).float()
        if cuda:
            noisy_images, original_images = noisy_images.cuda(), original_images.cuda()
    
        DLoader = DataLoader(dataset=list(zip(noisy_images, original_images)), batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        # 遍历每个批次
        for batch_noisy, batch_original in DLoader:
            optimizer.zero_grad()
            if cuda:
                batch_noisy, batch_original = batch_noisy.cuda(), batch_original.cuda()

            # 计算损失并更新参数
            loss = criterion(model(batch_noisy), batch_original)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()
        elapsed_time = time.time() - start_time

        # 输出每个 epoch 的损失
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(DLoader):.4f}, Time: {elapsed_time:.2f}s')
        if (epoch+1)%30==0:
            torch.save(model, os.path.join(save_dir, 'final_model.pth'))
            print(f"Epoch {epoch + 1}训练结束，模型已保存至 {os.path.join(save_dir, 'final_model.pth')}")
    # 训练结束后保存模型
    torch.save(model, os.path.join(save_dir, 'final_model.pth'))
    print(f"训练结束，模型已保存至 {os.path.join(save_dir, 'final_model.pth')}")
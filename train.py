# 时间： 2024/11/14 19:13
# -*- coding: utf-8 -*-

import os
import torch
from data import My_Model_Dataset
from option import args
from model import DnCNN  # 我模型还没改，写的DnCNN

batch_size = args.batch_size
n_epoch = args.epoch
lr = args.lr
save_dir = os.path.join('models', 'lj_oct')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
cuda = torch.cuda.is_available()  # 检查是否有可用的 GPU

if __name__ == '__main__':
    # 加载数据
    train_data = My_Model_Dataset(args, args.dir_data)
    print(len(train_data))
    # 加载模型
    Model = DnCNN(args)
    print("能不能用gpu:",torch.cuda.is_available())
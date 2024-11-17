# 时间： 2024/11/14 19:13
import os
import time
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from data import My_Model_Dataset
from option import args
from model import DnCNN  # 我模型还没改，写的DnCNN
from loss import SSIM, MS_SSIM
from test import test

save_dir = os.path.join('models', 'lj_oct')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


def train(args):
    # ---------------------------------------configuration---------------------------------------------------
    # 根据是否有可用的GPU来选择计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置批量大小
    batch_size = args.batch_size

    # ====================================step 1/5: 数据准备========================================================
    # 构建测试数据集实例
    test_data = My_Model_Dataset(args, args.dir_test_ori_img, args.dir_test_noi_img, mode='test')

    # 构建测试数据加载器
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    # ===================================step 2/5: 模型构建========================================================
    # 实例化模型对象
    _model = DnCNN(args)
    _model = _model.to(device)

    # ====================================step 3/5: 损失函数定义================================================
    # 根据命令行参数选择合适的损失函数
    if args.loss_func.lower() == 'l2':
        criterion = nn.MSELoss(reduction='sum')  # 使用均方误差损失,reduction可以改成mean算均值，但是这样损失显示出来就很小
    elif args.loss_func.lower() == 'ms-ssim':
        criterion = MS_SSIM()  # 使用多尺度结构相似性损失
    elif args.loss_func.lower() == 'ssim':
        criterion = SSIM()  # 使用多尺度结构相似性损失
    elif args.loss_func.lower() == 'l1':
        criterion = nn.L1Loss(reduction='sum')  # 使用绝对误差损失
    elif args.loss_func.lower() == 'smooth':
        criterion = nn.SmoothL1Loss(reduction='sum')  # 使用平滑L1损失
    else:
        raise ValueError("Please input the correct loss function with --loss_func $loss function(mse or ssim)...")

    # ====================================step 4/5: 优化器定义================================================
    # 定义Adam优化器
    optimizer = optim.Adam(_model.parameters(), lr=args.lr)

    # 如果指定了起始轮次大于0，则加载之前训练的状态
    if args.start_epoch > 0:
        print("Start to load state from %d epoch.............." % args.start_epoch)

        # 构建状态文件路径
        state_path = os.path.join(save_dir, 'model_%03d.pth' % args.start_epoch)
        # 加载状态文件
        checkpoint = torch.load(state_path)
        _model.model.load_state_dict(checkpoint['net'])  # 加载模型参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器状态
        start_epoch = checkpoint['epoch']  # 设置起始轮次
    else:
        start_epoch = 0  # 如果没有指定起始轮次，则从第0轮开始

    # 定义学习率调度器
    milestone = [15 - args.start_epoch,
                 22 - args.start_epoch,
                 24 - args.start_epoch,
                 26 - args.start_epoch,
                 28 - args.start_epoch,
                 30 - args.start_epoch,
                 31 - args.start_epoch,
                 190]

    milestone = list(filter(lambda x: x > 0, milestone))  # 过滤掉负数里程碑

    scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=0.5)  # 学习率调度器

    # ====================================step 5/5: 训练过程================================================
    for epoch in range(start_epoch, args.epoch):
        start_time = time.time()
        _model.train()
        # 构建训练数据集实例
        train_data = My_Model_Dataset(args, image_dir= args.dir_data, mode='train')
        # 构建训练数据加载器
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=1)

        # 训练
        loss_sigma = []
        for n_count, data in enumerate(train_loader):
            # 取出数据和标签
            ori_img, nos_img = data
            ori_img, nos_img = ori_img.to(device), nos_img.to(device)
            # 前向传播
            outputs = _model(nos_img)
            optimizer.zero_grad()
            # 计算损失函数
            loss = criterion(outputs, ori_img)
            loss.backward()
            optimizer.step()
            loss_sigma.append(loss.item())

        epoch_loss = np.mean(loss_sigma)
        scheduler.step()
        elapsed_time = time.time() - start_time

        # 打印当前轮次的训练损失和学习率、时间
        print("=========Epoch[{:0>3}/{:0>3}]  Train loss:{:.4f}  LR:{}  Time:{:.2f}s=============".format(
            epoch + 1, args.epoch, epoch_loss, optimizer.param_groups[0]["lr"], elapsed_time))

        # ======================================保存模型和状态================================================
        # 每隔十轮保存一次模型和状态
        if (epoch + 1) % args.save_model_epoch == 0:
            model_save_path = os.path.join(save_dir, f'model_epoch_{epoch + 1:03d}.pth')
            torch.save(_model, model_save_path)

            # 在测试数据集上评估模型性能
            psnr_avg, ssim_avg, _loss = test(args,
                                             test_loader,
                                             args.save_test_dir,
                                             save=False,
                                             model_file=_model,
                                             loss_f=criterion)

            print("Epoch: {},  Loss: {:.4f}, PSNR: {:.4f},  SSIM: {:.4f}, Test Loss: {:.4f}".format(
                 epoch + 1, epoch_loss, psnr_avg, ssim_avg, _loss))

    torch.save(_model, os.path.join(save_dir, 'final_model.pth'))
    print(f"训练结束，模型已保存至 {os.path.join(save_dir, 'final_model.pth')}")
    print("使用的是", str(device))


if __name__ == '__main__':
    print("能不能用gpu:", torch.cuda.is_available())
    train(args)


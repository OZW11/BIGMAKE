# 时间： 2024/11/14 19:13
import os
import time
import numpy as np
import torch
from torch import nn, optim
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from data import My_Art_Dataset, My_Real_Dataset
from option import args
import model   
from loss import SSIM, MS_SSIM
from test import test

save_dir = args.model_dir
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# 设置训练模式，使用人工生成噪声 or 采集噪声
if args.training_mode == 'art':  # 人工生成噪声
    My_Train_Dataset = My_Art_Dataset
elif args.training_mode == 'real':  # 采集噪声
    My_Train_Dataset = My_Real_Dataset
else:
    raise ValueError("args.training_mode must be art or real")

# 设置测试模式，使用人工生成噪声 or 采集噪声
if args.testing_mode == 'art':  # 人工生成噪声
    My_Test_Dataset = My_Art_Dataset
elif args.testing_mode == 'real':  # 采集噪声
    My_Test_Dataset = My_Real_Dataset
else:
    raise ValueError("args.testing_mode must be art or real")


def train(args):
    # ---------------------------------------configuration---------------------------------------------------
    # 根据是否有可用的GPU来选择计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置批量大小
    batch_size = args.batch_size

    # ====================================step 1/5: 数据准备========================================================
    # 构建测试数据集实例
    test_data = My_Test_Dataset(args, image_dir=args.dir_test_ori_img, noise_dir=args.dir_test_noi_img, mode='test')

    # 构建测试数据加载器
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    # ===================================step 2/5: 模型构建========================================================
    # 实例化模型对象
    _model = model.FilterNet(args)
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
    # 根据命令行参数选择合适的优化器
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(_model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(_model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(_model.parameters(), lr=args.lr)
    else:
        raise ValueError("Please input the correct optimizer with --optimizer $optimizer(Adam or SGD or RMSprop)...")

    # ====================================学习率调度器========================================================
    if args.scheduler.lower() == 'steplr':
        step_size = args.step_size if hasattr(args, 'step_size') else 100
        gamma = args.gamma if hasattr(args, 'gamma') else 0.7
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif args.scheduler.lower() == 'cosineannealinglr':
        T_max = args.t_max if hasattr(args, 't_max') else 100
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=args.lr_min)

    else:
        raise ValueError(
            "请正确选择学习率调度器：--scheduler $scheduler(steplr, cosineannealinglr)...")

    # 如果指定了起始轮次大于0，则加载之前训练的状态
    if args.start_epoch > 0:
        print("Start to load state from %d epoch.............." % args.start_epoch)

        # 构建状态文件路径
        state_path = os.path.join(save_dir, 'model_epoch_%03d.pth' % args.start_epoch)
        # 加载状态文件
        checkpoint = torch.load(state_path, weights_only=False)  # 显式声明使用完整状态
        _model.load_state_dict(checkpoint['model_state'])  # 加载模型参数
        optimizer.load_state_dict(checkpoint['optimizer_state'])  # 加载优化器状态
        scheduler.load_state_dict(checkpoint['scheduler_state'])  # 加载调度器状态
        start_epoch = checkpoint['epoch']  # 恢复训练轮次
    else:
        start_epoch = 0  # 如果没有指定起始轮次，则从第0轮开始

    # ====================================step 5/5: 训练过程================================================
    for epoch in range(start_epoch, args.epoch):
        start_time = time.time()
        _model.train()
        # 构建训练数据集实例
        train_data = My_Train_Dataset(args, image_dir=args.dir_train_ori_img, noise_dir=args.dir_train_noi_img,
                                      mode='train')
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
        # 每隔多少轮保存一次模型和状态
        if (epoch + 1) % args.save_model_epoch   == 0:
            model_save_path = os.path.join(save_dir, f'model_epoch_{epoch + 1:03d}.pth')
            torch.save({
                'epoch': epoch,
                'model_state': _model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict()  # 保存调度器的状态
            }, model_save_path)

            #在测试数据集上评估模型性能
            # psnr_avg, ssim_avg, _loss = test(args,
            #                                 test_loader,
            #                                 args.save_test_dir,
            #                                 save=False,
            #                                 model_file=_model,
            #                                 loss_f=criterion,
            #                                 device=device)

            # print("Epoch: {},  Loss: {:.4f}, PSNR: {:.4f},  SSIM: {:.4f}, Test Loss: {:.4f}".format(
            #     epoch + 1, epoch_loss, psnr_avg, ssim_avg, _loss))

    torch.save({
        'epoch': epoch,
        'model_state': _model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict()
    }, os.path.join(save_dir, 'final_model.pth'))
    print(f"训练结束，模型已保存至 {os.path.join(save_dir, 'final_model.pth')}")
    print("使用的是", str(device))   


if __name__ == '__main__':
    print("能不能用gpu:", torch.cuda.is_available())
    print("使用的训练数据是：", args.training_mode, "使用的测试数据是：", args.testing_mode)
    train(args)

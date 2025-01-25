# 时间： 2024/11/14 20:44
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import nn
from torch.utils.data import DataLoader
from data import My_Art_Dataset, My_Real_Dataset
from option import args
import model


def test(args, data_loader, save_test_dir, device, save=False, model_file=None, loss_f=None):
    """
    参数:
        args: 命令行参数对象
        test_loader: 测试数据加载器
        save_test_dir: 可选，保存测试结果的目录
        save: 是否保存去噪结果
        model_file: 测试模型
        loss_f: 损失函数对象

    返回:
        psnr_avg: 平均 PSNR
        ssim_avg: 平均 SSIM
        loss_avg: 平均损失
    """
    # 初始化性能评估指标
    psnr_values, ssim_values, loss_values = [], [], []

    with torch.no_grad():
        for idx, (ori_img, nos_img, img_name) in enumerate(data_loader):
            # 将含有噪声的图像移动到指定设备上
            nos_img = nos_img.to(device)
            # 模型去噪
            output = model_file(nos_img)

            # 计算损失
            ori_img = ori_img.to(device)
            loss = loss_f(output, ori_img).item()
            loss_values.append(loss)

            # 数据范围还原
            ori_img_np = ori_img.mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8)
            nos_img_np = nos_img.mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8)
            output_np = output.mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8)

            # 去除 batch 维度
            ori_img_np = np.squeeze(ori_img_np)
            nos_img_np = np.squeeze(nos_img_np)
            output_np = np.squeeze(output_np)

            # 计算 PSNR 和 SSIM
            psnr = peak_signal_noise_ratio(ori_img_np, output_np, data_range=255)
            ssim = structural_similarity(ori_img_np, output_np, data_range=255, multichannel=False)
            psnr_values.append(psnr)
            ssim_values.append(ssim)

            if save:
                save_path = os.path.join(save_test_dir, f"{img_name}_comparison.png")
            else:
                save_path = None
            nos_img_np = nos_img.cpu().squeeze().numpy()  # 转换为 Numpy 格式
            ori_img_np = ori_img.cpu().squeeze().numpy()
            output_np = output.cpu().squeeze().numpy()
            show_and_save_comparison(args, nos_img_np, ori_img_np, output_np, save_path)

    # 计算平均性能指标
    psnr_avg = np.mean(psnr_values)
    ssim_avg = np.mean(ssim_values)
    loss_avg = np.mean(loss_values)

    print(f"\n测试结果: PSNR Avg={psnr_avg:.2f}, SSIM Avg={ssim_avg:.4f}, Loss Avg={loss_avg:.4f}")

    return psnr_avg, ssim_avg, loss_avg


def show_and_save_comparison(aegs, noisy, original, denoised, save_path=None):
    """
    显示并保存对比图像

    参数:
        noisy: 含噪声的图像（numpy array）
        original: 原始图像（numpy array）
        denoised: 去噪后的图像（numpy array）
        save_path: 保存路径，若为 None 则不保存
    """
    # 创建对比图像
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(noisy, cmap='gray')
    if args.testing_mode == 'art':
        plt.title(f'{args.noise_type} Noisy Image')
    elif args.testing_mode == 'real':
        plt.title('Collecting Noisy Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(denoised, cmap='gray')
    plt.title('Denoised Image')
    plt.axis('off')

    # 保存和显示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"保存图像至: {save_path}")
    plt.show()


if __name__ == '__main__':
    print("能不能用gpu:", torch.cuda.is_available())
    print("Start to test.......")

    # 设置测试模式，使用人工生成噪声 or 采集噪声
    if args.testing_mode == 'art':  # 人工生成噪声
        My_Test_Dataset = My_Art_Dataset
    elif args.testing_mode == 'real':  # 采集噪声
        My_Test_Dataset = My_Real_Dataset
    else:
        raise ValueError("args.testing_mode must be art or real")

    test_data = My_Test_Dataset(args, image_dir=args.dir_test_ori_img, noise_dir=args.dir_test_noi_img, mode='test')
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    # 定义损失函数，使用均方误差损失（MSE）
    criterion = nn.MSELoss(reduction='sum')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    _model = model.focusFilter(args).to(device)

    # 加载保存的模型文件
    checkpoint = torch.load(os.path.join(args.model_dir, args.model_name), map_location=device)

    # 区分保存方式
    if isinstance(checkpoint, dict):  # 如果是字典，说明保存的是 state_dict
        print("检测到保存的是 state_dict 格式")

        # 初始化模型
        _model = model.focusFilter(args).to(device)

        # 加载 state_dict
        if "model_state" in checkpoint:  # 检查是否包含键
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint  # 直接是 state_dict 本身

        _model.load_state_dict(state_dict)

    elif isinstance(checkpoint, nn.Module):  # 如果是 nn.Module，说明保存的是整个模型实例
        print("检测到保存的是模型实例")
        _model = checkpoint.to(device)

    else:
        raise TypeError("未知的模型文件格式，请检查保存方式！")

    # 切换模型到评估模式
    _model.eval()

    # 调用测试函数
    psnr_avg, ssim_avg, _loss = test(args,
                                     test_loader,
                                     args.save_test_dir,
                                     device,
                                     save=False,
                                     model_file=_model,  # 传入正确的模型实例
                                     loss_f=criterion)
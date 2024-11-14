# 时间： 2024/11/14 20:44
import torch.nn as nn


# 定义DnCNN模型
class DnCNN(nn.Module):
    def __init__(self, args, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(image_channels, n_channels, kernel_size=kernel_size, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size=kernel_size, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise

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
    

# 定义 Ures 模型
class Ures(nn.Module):
    def __init__(
        self,
        args,
        image_channels=1,
    ):
        super(Ures, self).__init__()

        layers1 = []
        layers2 = []
        layers3 = []
        layers4 = []
        layers5 = []
        layers6=[]
        
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
        layers1.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers1.append(nn.ReLU(inplace=True))

        layers2.append(nn.MaxPool2d(2))
        layers2.append(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False
            )
        )
        layers2.append(nn.BatchNorm2d(128, eps=0.0001, momentum=0.90))
        layers2.append(nn.ReLU(inplace=True))

        layers3.append(nn.MaxPool2d(2))
        layers3.append(
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False
            )
        )
        layers3.append(nn.BatchNorm2d(128, eps=0.0001, momentum=0.90))
        layers3.append(nn.ReLU(inplace=True))

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
        layers5.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers5.append(nn.ReLU(inplace=True))
        
        layers6.append(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False
            )
        )
        layers6.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers6.append(nn.ReLU(inplace=True))
        layers6.append(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False
            )
        )
        layers6.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers6.append(nn.ReLU(inplace=True))
        layers6.append(
            nn.Conv2d(
                in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=False
            )
        )

        self._1Out = nn.Sequential(*layers1)
        self._2Out = nn.Sequential(*layers2)
        self._3Out = nn.Sequential(*layers3)
        self._4Out = nn.Sequential(*layers4)
        self._5Out = nn.Sequential(*layers5)
        self._6Out = nn.Sequential(*layers6)
        
        self._initialize_weights()

    def forward(self, x):
        y = x
        x1 = self._1Out(x)
        x2 = self._2Out(x1)
        x3 = self._3Out(x2)
        x3 = nn.functional.interpolate(x3, size=(int(x.shape[2]/2), int(x.shape[3]/2)), mode="nearest")
        x4 = self._4Out(x3)
        x4 = nn.functional.interpolate(x4, size=(int(x.shape[2]), int(x.shape[3])), mode="nearest")
        x5 = self._5Out(x4)
        x5 = x1-x5
        x6 = self._6Out(x5)
        return y - x6

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, nonlinearity="relu"
                )  # 使用 Kaiming 初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



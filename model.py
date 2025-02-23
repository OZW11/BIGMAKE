# 时间： 2024/11/14 20:44
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch

# 定义DnCNN模型
class DnCNN(nn.Module):
    def __init__(
        self,
        args,
        depth=17,
        n_channels=64,
        image_channels=1,
        use_bnorm=True,
        kernel_size=3,
    ):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                image_channels,
                n_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(
                    n_channels,
                    n_channels,
                    kernel_size=kernel_size,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(
                n_channels,
                image_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
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
        layers6 = []

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
        layers1.append(nn.Tanh())
        layers1.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))


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
        layers4.append(nn.ReLU(inplace=True))
        layers4.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
      

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
        x3 = nn.functional.interpolate(
            x3, size=(int(x.shape[2] / 2), int(x.shape[3] / 2)), mode="nearest"
        )
        x4 = self._4Out(x3)
        x4 = nn.functional.interpolate(
            x4, size=(int(x.shape[2]), int(x.shape[3])), mode="nearest"
        )
        x5 = self._5Out(x4)
        x5 = x1 - x5
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


# 定义 PDnet 模型
class PDnet(nn.Module):
    def __init__(
        self,
        args,
        image_channels=1,
    ):
        super(PDnet, self).__init__()

        layers1 = []
        layers2 = []
        layers3 = []
        layers4 = []
        layers5 = []
        layers6 = []

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
        layers1.append(
            nn.Conv2d(
                in_channels=64,
                out_channels=1,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True,
            )
        )



        layers2.append(
            nn.Conv2d(
                in_channels=2, out_channels=64, kernel_size=3, padding=1, bias=False
            )
        )
        layers2.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers2.append(nn.ReLU(inplace=True))

        layers3.append(nn.MaxPool2d(2))
        layers3.append(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False
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

        layers4.append(
            nn.Conv2d(
                in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=False
            )
        )
        layers4.append(nn.ReLU(inplace=True))

        layers5.append(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False
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
        x1 = self._1Out(x)*10
        x1_=(x1-x1.min())/(x1.max()-x1.min())
        x1 = torch.log10(x1_+0.5)*4096*2
        x1__=torch.cat((x,x1_),1)
        x2 = self._2Out(x1__)
        x3 = self._3Out(x2)
        x3 = nn.functional.interpolate(
            x3, size=(int(x.shape[2] ), int(x.shape[3] )), mode="nearest"
        )
        x4 = self._4Out(x3)
        x4=x-x4
        x5 = self._5Out(x4)
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



# 定义 Ures 模型
class Ures_(nn.Module):
    def __init__(
        self,
        args,
        image_channels=1,
    ):
        super(Ures_, self).__init__()

        layers1 = []
        layers2 = []
        layers3 = []
        layers4 = []
        layers5 = []
        layers6 = []

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
        layers1.append(nn.Tanh())
        layers1.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))


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
        layers4.append(nn.ReLU(inplace=True))
        layers4.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
      

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

        layers1_ = []
        layers2_ = []
        layers3_ = []
        layers4_ = []
        layers5_ = []
        layers6_ = []

        layers1_.append(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True,
            )
        )
        layers1_.append(
            nn.Conv2d(
                in_channels=64,
                out_channels=1,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True,
            )
        )

        layers2_.append(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False
            )
        )
        layers2_.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers2_.append(nn.ReLU(inplace=True))#之前少加了_

        layers3_.append(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False
            )
        )
        layers3_.append(nn.BatchNorm2d(128, eps=0.0001, momentum=0.90))
        layers3_.append(nn.ReLU(inplace=True))

        layers4_.append(
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False
            )
        )
        layers4_.append(nn.ReLU(inplace=True))
        layers4_.append(
            nn.Conv2d(
                in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False
            )
        )
        layers4_.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers4_.append(nn.ReLU(inplace=True))

        layers4_.append(
            nn.Conv2d(
                in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=False
            )
        )
        layers4_.append(nn.ReLU(inplace=True))

        layers5_.append(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False
            )
        )
        layers5_.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers5_.append(nn.ReLU(inplace=True))

        layers6_.append(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False
            )
        )
        layers6_.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers6_.append(nn.ReLU(inplace=True))
        layers6_.append(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False
            )
        )
        layers6_.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers6_.append(nn.ReLU(inplace=True))
        layers6_.append(
            nn.Conv2d(
                in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=False
            )
        )

        self._1Outs = nn.Sequential(*layers1_)
        self._2Outs = nn.Sequential(*layers2_)
        self._3Outs = nn.Sequential(*layers3_)
        self._4Outs = nn.Sequential(*layers4_)
        self._5Outs = nn.Sequential(*layers5_)
        self._6Outs = nn.Sequential(*layers6_)


        self._initialize_weights()

    def forward(self, x):
        
        y = x
        x1 = self._1Out(x)
        x2 = self._2Out(x1)
        x3 = self._3Out(x2)
        x3 = nn.functional.interpolate(
            x3, size=(int(x.shape[2] / 2), int(x.shape[3] / 2)), mode="nearest"
        )
        x4 = self._4Out(x3)
        x4 = nn.functional.interpolate(
            x4, size=(int(x.shape[2]), int(x.shape[3])), mode="nearest"
        )
        x5 = self._5Out(x4)
        x5 = x1 - x5
        x6 = self._6Out(x5)
        
        
        x1s = self._1Outs(x)
        x2s = self._2Outs(x1s)
        x3s = self._3Outs(x2s)
        x4s = self._4Outs(x3s)
        x4s=x-x4s
        x5s = self._5Outs(x4s)
        x6s = self._6Outs(x5s)
        return y-(x6s+x6)/2
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


# 定义LoopyNet模型
class LoopyNet(nn.Module):
    def __init__(
        self,
        args,
        image_channels=1,
        use_bnorm=True,
        kernel_size=3,
    ):
        super(LoopyNet, self).__init__()
        
        layers_in = []
        layers_loop1_32to64 = []
        layers_loop1_64to32 = []
        layers_loop2_32to64 = []
        layers_loop2_64to64 = []
        layers_loop2_64to32 = []
        layers_out = []
#******************************************************#
        layers_in.append(
            nn.Conv2d(
                image_channels,
                64,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers_in.append(nn.ReLU(inplace=True))
        
        layers_loop1_32to64.append(
            nn.Conv2d(
                64,
                128,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers_loop1_32to64.append(nn.BatchNorm2d(128, eps=0.0001, momentum=0.90))
        layers_loop1_32to64.append(nn.ReLU(inplace=True))
        
        layers_loop1_64to32.append(
            nn.Conv2d(
                128,
                64,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers_loop1_64to32.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers_loop1_64to32.append(nn.ReLU(inplace=True))

        layers_loop2_32to64.append(
            nn.Conv2d(
                64,
                128,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers_loop2_32to64.append(nn.BatchNorm2d(128, eps=0.0001, momentum=0.90))
        layers_loop2_32to64.append(nn.ReLU(inplace=True))

        layers_loop2_64to64.append(
            nn.Conv2d(
                128,
                128,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers_loop2_64to64.append(nn.BatchNorm2d(128, eps=0.0001, momentum=0.90))
        layers_loop2_64to64.append(nn.ReLU(inplace=True))

        layers_loop2_64to32.append(
            nn.Conv2d(
                128,
                64,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers_loop2_64to32.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers_loop2_64to32.append(nn.ReLU(inplace=True))
        
        layers_out.append(
            nn.Conv2d(
                32,
                image_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            ))
        
        layers2_ = []
        layers3_ = []
        layers4_ = []
        layers5_ = []
        layers6_ = []


        layers2_.append(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False
            )
        )
        layers2_.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers2_.append(nn.ReLU(inplace=True))

        layers3_.append(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False
            )
        )
        layers3_.append(nn.BatchNorm2d(128, eps=0.0001, momentum=0.90))
        layers3_.append(nn.ReLU(inplace=True))

        layers4_.append(
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False
            )
        )
        layers4_.append(nn.ReLU(inplace=True))
        layers4_.append(
            nn.Conv2d(
                in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False
            )
        )
        layers4_.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers4_.append(nn.ReLU(inplace=True))

        layers4_.append(
            nn.Conv2d(
                in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=False
            )
        )
        layers4_.append(nn.ReLU(inplace=True))

        layers5_.append(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False
            )
        )
        layers5_.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers5_.append(nn.ReLU(inplace=True))

        layers6_.append(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False
            )
        )
        layers6_.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers6_.append(nn.ReLU(inplace=True))
        layers6_.append(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False
            )
        )
        layers6_.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers6_.append(nn.ReLU(inplace=True))
        layers6_.append(
            nn.Conv2d(
                in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=False
            )
        )


        
#******************************************************#
        self.deelInput = nn.Sequential(*layers_in)
        self.deel_loop1_32to64 = nn.Sequential(*layers_loop1_32to64)
        self.deel_loop1_64to32 = nn.Sequential(*layers_loop1_64to32)
        self.deel_loop2_32to64 = nn.Sequential(*layers_loop2_32to64)
        self.deel_loop2_64to64 = nn.Sequential(*layers_loop2_64to64)
        self.deel_loop2_64to32 = nn.Sequential(*layers_loop2_64to32)
        self.deelOUT = nn.Sequential(*layers_out)
        
        self._2Out = nn.Sequential(*layers2_)
        self._3Out = nn.Sequential(*layers3_)
        self._4Out = nn.Sequential(*layers4_)
        self._5Out = nn.Sequential(*layers5_)
        self._6Out = nn.Sequential(*layers6_)

    def forward(self, x):
        orin=x
        IN1 = self.deelInput(x)
        
        Out_loop2= IN1
        loop2_lastin=IN1-IN1
        for i in range(1):      
            loop2_A=Out_loop2-loop2_lastin
            loop2_lastin=Out_loop2
            
            Out_loop1= loop2_A
            loop1_lastin=loop2_A-loop2_A
            for j in range(3):
                loop1_A=Out_loop1-loop1_lastin
                loop1_lastin=Out_loop1
                loop1_B=self.deel_loop1_32to64(loop1_A)
                Out_loop1=self.deel_loop1_64to32(loop1_B)
            
            loop2_B=IN1-Out_loop1
            loop2_C=self.deel_loop2_32to64(loop2_B)
            loop2_D=self.deel_loop2_64to64(loop2_C)
            Out_loop2=self.deel_loop2_64to32(loop2_D)

        x2 = self._2Out(Out_loop2)
        x3 = self._3Out(x2)
        x3 = nn.functional.interpolate(
            x3, size=(int(x.shape[2] / 2), int(x.shape[3] / 2)), mode="nearest"
        )
        x4 = self._4Out(x3)
        x4 = nn.functional.interpolate(
            x4, size=(int(x.shape[2]), int(x.shape[3])), mode="nearest"
        )
        x5 = self._5Out(x4)
        x5 = x2 - x5
        x6 = self._6Out(x5)

        return orin - x6

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



# 定义focusFilter模型
class focusFilter(nn.Module):
    def __init__(
        self,
        args,
        n_channels=64,
        image_channels=1,
        use_bnorm=True,
        kernel_size=3,
    ):
        super(focusFilter, self).__init__()
        layers1 = []
        layers1.append(
            nn.Conv2d(
                image_channels,
                n_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers1.append(nn.ReLU(inplace=True))
        self.deel1 = nn.Sequential(*layers1)
        
        layers2 = []
        layers2_ = []
        layers2.append(
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers2.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers2.append(nn.ReLU(inplace=True))
        layers2.append(
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers2.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers2.append(nn.ReLU(inplace=True))
        layers2.append(nn.MaxPool2d(2))
        layers2_.append(
            nn.Conv2d(
                n_channels,
                (int)(n_channels/2),
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers2_.append(
            nn.Conv2d(
                (int)(n_channels/2),
                n_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers2_.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers2_.append(nn.ReLU(inplace=True))
        self.deel2 = nn.Sequential(*layers2)
        self.deel2_ = nn.Sequential(*layers2_)
        
        layers3 = []
        layers3_ = []
        layers3.append(
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers3.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers3.append(nn.ReLU(inplace=True))
        layers3.append(
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers3.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers3.append(nn.ReLU(inplace=True))
        layers3.append(nn.MaxPool2d(2))
        layers3_.append(
            nn.Conv2d(
                n_channels,
                (int)(n_channels/2),
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers3_.append(
            nn.Conv2d(
                (int)(n_channels/2),
                n_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers3_.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers3_.append(nn.ReLU(inplace=True))
        self.deel3 = nn.Sequential(*layers3)
        self.deel3_ = nn.Sequential(*layers3_)
        
        layers4 = []
        layers4_ = []
        layers4.append(
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers4.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers4.append(nn.ReLU(inplace=True))
        layers4.append(
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers4.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers4.append(nn.ReLU(inplace=True))
        layers4.append(nn.AvgPool2d(2))
        layers4_.append(
            nn.Conv2d(
                n_channels,
                (int)(n_channels/2),
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers4_.append(
            nn.Conv2d(
                (int)(n_channels/2),
                n_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers4_.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers4_.append(nn.ReLU(inplace=True))
        self.deel4 = nn.Sequential(*layers4)
        self.deel4_ = nn.Sequential(*layers4_)
        
        layers5 = []
        layers5_ = []
        layers5.append(
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers5.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers5.append(nn.ReLU(inplace=True))
        layers5.append(
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers5.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers5.append(nn.ReLU(inplace=True))
        layers5_.append(
            nn.Conv2d(
                n_channels,
                (int)(n_channels/2),
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers5_.append(
            nn.Conv2d(
                (int)(n_channels/2),
                n_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers5_.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers5_.append(nn.ReLU(inplace=True))
        self.deel5 = nn.Sequential(*layers5)
        self.deel5_ = nn.Sequential(*layers5_)
        
        layers6 = []
        layers6.append(
            nn.Conv2d(
                (int)(n_channels*5),
                n_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers6.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers6.append(nn.ReLU(inplace=True))
        layers6.append(
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        
        layers6.append(nn.BatchNorm2d(64, eps=0.0001, momentum=0.90))
        layers6.append(nn.ReLU(inplace=True))
        layers6.append(
            nn.Conv2d(
                n_channels,
                (int)(n_channels/2),
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        layers6.append(
            nn.Conv2d(
                (int)(n_channels/2),
                image_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            )
        )
        self.deel6 = nn.Sequential(*layers6)
        
###############  focus system  #####################
        flayers1 = []
        flayers1.append(
            nn.Conv2d(
                image_channels,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            )
        )
        flayers1.append(nn.ReLU(inplace=True))
        self.fdeel1 = nn.Sequential(*flayers1)
        
        flayers2 = []
        flayers2.append(
            nn.Conv2d(
                64,
                image_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
        )
        flayers2.append(nn.ReLU(inplace=True))
        self.fdeel2 = nn.Sequential(*flayers2)
        

    def forward(self, x):
        x1 = self.deel1(x)
        focusmatrix = self.fdeel1(x)
        focusmatrix = self.fdeel2(focusmatrix)
        
        x2 = self.deel2(x1)
        x2 = self.deel2_(x2)
        x3 = self.deel3(x1)
        x3 = self.deel3_(x3)
        x4 = self.deel4(x1)
        x4 = self.deel4_(x4)
        x5 = self.deel5(x1)

        
        x2 = nn.functional.interpolate(
            x2, size=(int(x.shape[2]), int(x.shape[3])), mode="nearest"
        )
        x3 = nn.functional.interpolate(
            x3, size=(int(x.shape[2]), int(x.shape[3])), mode="nearest"
        )
        x4 = nn.functional.interpolate(
            x4, size=(int(x.shape[2]), int(x.shape[3])), mode="nearest"
        )

        
        tensors=[x2,x3,x4,x5,x1]
        xcat = torch.cat(tensors, dim=1) 
        x6 = x-self.deel6(xcat)*focusmatrix
        return x6

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * out.sigmoid().view(b, c, 1, 1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels, eps=0.0001, momentum=0.1),  # 修正momentum
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels, eps=0.0001, momentum=0.1),
            ChannelAttention(in_channels)  # 通道注意力
        )
        
    def forward(self, x):
        return x + self.conv(x)

class FilterNet(nn.Module):
    def __init__(
        self,
        args,
        n_channels=64,
        image_channels=1,
        use_bnorm=True,
        kernel_size=3,
    ):
        super(FilterNet, self).__init__()
        
        # 初始卷积层
        self.initial = nn.Sequential(
            nn.Conv2d(image_channels, n_channels, kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 下采样路径
        self.down1 = nn.Sequential(
            *[ResidualBlock(n_channels) for _ in range(2)],
            nn.Conv2d(n_channels, n_channels*2, 3, stride=2, padding=1, bias=False)
        )
        
        self.down2 = nn.Sequential(
            *[ResidualBlock(n_channels*2) for _ in range(2)],
            nn.Conv2d(n_channels*2, n_channels*4, 3, stride=2, padding=1, bias=False)
        )

        # 中间特征处理
        self.mid = nn.Sequential(
            *[ResidualBlock(n_channels*4) for _ in range(4)],
            ChannelAttention(n_channels*4)
        )

        # 上采样路径
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(n_channels*4, n_channels*2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(n_channels*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(n_channels*2*2, n_channels, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(n_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 特征融合模块
        self.fusion = nn.Sequential(
            ResidualBlock(n_channels*2),
            nn.Conv2d(n_channels*2, n_channels, 1, bias=False),
            ChannelAttention(n_channels)
        )

        # 最终重建层
        self.final = nn.Sequential(
            *[ResidualBlock(n_channels) for _ in range(2)],
            nn.Conv2d(n_channels, image_channels, kernel_size, padding=1, bias=False)
        )

    def forward(self, x):
        x0 = self.initial(x)  # 初始特征
        
        # 编码路径
        d1 = self.down1(x0)    # 1/2分辨率
        d2 = self.down2(d1)    # 1/4分辨率
        
        # 中间处理
        mid = self.mid(d2)
        
        # 解码路径
        u1 = self.up1(mid)     # 1/2分辨率
        u1 = torch.cat([u1, d1], dim=1)
        
        u2 = self.up2(u1)      # 原始分辨率
        u2 = torch.cat([u2, x0], dim=1)
        
        # 特征融合
        fused = self.fusion(u2)
        
        # 残差学习
        return self.final(fused)

#//////////////////////////////////////////////////////////#

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels=1, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # 输入形状: (B, C, H, W)
        x = self.proj(x)  # 输出形状: (B, embed_dim, H//p, W//p)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # 形状变为: (B, num_patches, embed_dim)
        return x, H, W

class TransformerDenoiser(nn.Module):
    def __init__(self, patch_size=16, in_channels=1, embed_dim=256, 
                 num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch嵌入层
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        
        # 位置编码（二维行列式编码）
        self.row_embed = nn.Parameter(torch.randn(50, embed_dim // 2))  # 最大支持50行
        self.col_embed = nn.Parameter(torch.randn(50, embed_dim // 2))  # 最大支持50列
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 解码器（重建像素值）
        self.decoder = nn.Linear(embed_dim, patch_size**2 * in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 分块嵌入
        x_patches, h, w = self.patch_embed(x)  # (B, N, D), H, W
        
        # 2. 位置编码
        row_emb = self.row_embed[:h]  # (h, D/2)
        col_emb = self.col_embed[:w]  # (w, D/2)
        
        # 组合行列编码
        pos_emb = torch.cat([
            row_emb.unsqueeze(1).expand(-1, w, -1),  # (h, w, D/2)
            col_emb.unsqueeze(0).expand(h, -1, -1)   # (h, w, D/2)
        ], dim=-1)  # (h, w, D)
        
        pos_emb = pos_emb.reshape(1, h*w, self.embed_dim)  # (1, N, D)
        x_patches = x_patches + pos_emb  # 加入位置信息
        
        # 3. Transformer编码
        encoded = self.transformer_encoder(x_patches)
        
        # 4. 重建像素
        decoded_patches = self.decoder(encoded)  # (B, N, patch_size²*C)
        
        # 5. 重组图像
        decoded_patches = decoded_patches.reshape(B, h, w, self.patch_size, self.patch_size)
        decoded_patches = decoded_patches.permute(0, 3, 1, 4, 2)  # (B, p, h, p, w)
        output = decoded_patches.reshape(B, 1, H, W)  # (B, C, H, W)
        
        return output
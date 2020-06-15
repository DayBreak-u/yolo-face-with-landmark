import torch
from torch import nn
import torch.nn.functional as F



class FPEM_fpn(nn.Module):
    def __init__(self, backbone_out_channels, **kwargs):
        """
        PANnet
        :param backbone_out_channels: 基础网络输出的维度
        """
        super().__init__()
        fpem_repeat = kwargs.get('fpem_repeat', 2)
        conv_out = 64
        # reduce layers

        self.reduce_conv_c3 = nn.Sequential(
            nn.Conv2d(in_channels=backbone_out_channels[0], out_channels=conv_out, kernel_size=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU()
        )
        self.reduce_conv_c4 = nn.Sequential(
            nn.Conv2d(in_channels=backbone_out_channels[1], out_channels=conv_out, kernel_size=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU()
        )
        self.reduce_conv_c5 = nn.Sequential(
            nn.Conv2d(in_channels=backbone_out_channels[2], out_channels=conv_out, kernel_size=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU()
        )
        self.fpems = nn.ModuleList()
        for i in range(fpem_repeat):
            self.fpems.append(FPEM(conv_out))


    def forward(self, x):
        c3, c4, c5 = x
        # reduce channel
        c3 = self.reduce_conv_c3(c3)
        c4 = self.reduce_conv_c4(c4)
        c5 = self.reduce_conv_c5(c5)

        # FPEM
        for i, fpem in enumerate(self.fpems):
            c3, c4, c5 = fpem(c3, c4, c5)
            if i == 0:
                c3_ffm = c3
                c4_ffm = c4
                c5_ffm = c5
            else:
                c3_ffm += c3
                c4_ffm += c4
                c5_ffm += c5


        return c3,c4,c5


class FPEM(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.up_add1 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add2 = SeparableConv2d(in_channels, in_channels, 1)
        self.down_add1 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add2 = SeparableConv2d(in_channels, in_channels, 2)


    def forward(self,  c3, c4, c5):
        # up阶段
        c4 = self.up_add1(self._upsample_add(c5, c4))
        c3 = self.up_add2(self._upsample_add(c4, c3))

        # down 阶段
        c4 = self.down_add1(self._upsample_add(c4, c3))
        c5 = self.down_add2(self._upsample_add(c5, c4))
        return  c3, c4, c5

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode="nearest") + y


class SeparableConv2d(nn.Module):
    def  __init__(self, in_channels, out_channels, stride=1):
        super(SeparableConv2d, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,
                                        stride=stride, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

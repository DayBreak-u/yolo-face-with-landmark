from torch import  nn
import torch
from torch.functional import  F
import numpy as np
from utils import  torch_utils
def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_bn1x1(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels,light_mode = False):
        super(FPN,self).__init__()

        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1)

        if light_mode:
            self.merge1 = conv_dw(out_channels,out_channels,1)
            self.merge2 = conv_dw(out_channels,out_channels,1)
        else:
            self.merge1 = conv_bn(out_channels, out_channels)
            self.merge2 = conv_bn(out_channels, out_channels)

    def forward(self, input):


        output1_ = self.output1(input[0])
        output2_ = self.output2(input[1])
        output3_ = self.output3(input[2])

        up3 = F.interpolate(output3_, size=[output2_.size(2), output2_.size(3)], mode="nearest")
        output2 = output2_ + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1_.size(2), output1_.size(3)], mode="nearest")
        output1 = output1_ + up2
        output1 = self.merge1(output1)

        # out = [output1, output2]
        out = [output1, output2, output3_]
        return out


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, stride,point_num=0,onnx_export=False):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.stride = stride  # layer stride
        self.na = len(anchors)    # number of anchors (3)
        self.nc = nc  # number of classes (1)
        self.no = nc + 5 + point_num * 2  # number of outputs (16)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.onnx_export = onnx_export
        self.point_num = point_num
        if self.onnx_export:
            self.training = False

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):


        if  self.onnx_export:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        if not  self.onnx_export:
            p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4 , 2).contiguous()  # prediction

        if self.training:
            return p

        elif  self.onnx_export:
            return p

        else:  # inference
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method

            for i in  range(self.point_num) :
                io[..., 4 + i *2:4+i*2 +2 ] = (io[..., 4 + i *2:4+i*2 +2 ] * self.anchor_wh) + self.grid

            io[..., :4+self.point_num*2] *= self.stride
            torch.sigmoid_(io[...,  self.point_num *2 +  4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]



class SPP(nn.Module):
    def __init__(self,inp):
        super(SPP, self).__init__()
        self.max_pool1 = nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=9,stride=1,padding=4)
        self.max_pool3 = nn.MaxPool2d(kernel_size=13,stride=1,padding=6)
        self.conv = conv_dw(inp*4,inp,1)

    def forward(self, x):
        pool1 = self.max_pool1(x)
        pool2 = self.max_pool2(x)
        pool3 = self.max_pool3(x)
        x = torch.cat((x,pool1,pool2,pool3),dim=1)
        x = self.conv(x)
        return x

class DarknetWithShh(nn.Module):
    # YOLOv3 object detection models

    def __init__(self, backbone , h ,  onnx_export = False, verbose = False , light_head = False ):
        super(DarknetWithShh, self).__init__()
        self.backbone = backbone
        self.onnx_export = onnx_export
        self.light_head = light_head
        feat_channel = self.backbone.feat_channel
        anchor =  h['anchors']
        out_channels = h['out_channels']
        anchor =  np.array(anchor)
        point_num =  h['point_num']
        nc =  h['nc']
        self.no = nc + 5 + point_num * 2
        self.fpn = FPN(feat_channel, out_channels , light_head)

        if self.light_head:
            self.light_head1 = nn.Sequential(conv_dw(out_channels,out_channels,1),
                                             nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels, bias=False),
                                             nn.BatchNorm2d(out_channels),
                                             nn.ReLU(inplace=True),
                                             )
            self.light_head2 = nn.Sequential(conv_dw(out_channels,out_channels,1),
                                             nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels, bias=False),
                                             nn.BatchNorm2d(out_channels),
                                             nn.ReLU(inplace=True),
                                             )
            self.light_head3 = nn.Sequential(conv_dw(out_channels,out_channels,1),
                                             nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels, bias=False),
                                             nn.BatchNorm2d(out_channels),
                                             nn.ReLU(inplace=True),
                                             )
        else:
            self.ssh1 = SSH(out_channels, out_channels)
            self.ssh2 = SSH(out_channels, out_channels)
            self.ssh3 = SSH(out_channels, out_channels)

        self.s8_head = nn.Conv2d(out_channels,len(anchor[0])  * self.no  ,kernel_size=(1,1),stride=1,padding=0)
        self.s16_head = nn.Conv2d(out_channels,len(anchor[1]) * self.no  ,kernel_size=(1,1),stride=1,padding=0)
        self.s32_head = nn.Conv2d(out_channels,len(anchor[2])  *  self.no  ,kernel_size=(1,1),stride=1,padding=0)

        self.yolo_s8  = YOLOLayer(anchor[0], nc, 8  , point_num=point_num , onnx_export=onnx_export)
        self.yolo_s16 = YOLOLayer(anchor[1], nc, 16 , point_num=point_num , onnx_export=onnx_export)
        self.yolo_s32 = YOLOLayer(anchor[2], nc, 32 , point_num=point_num , onnx_export=onnx_export)

        self.yolo_layers = self.get_yolo_layers()
        self.module_list = [m for (n,m) in self.named_modules()]




    def forward(self, x):
        out = self.backbone(x)


        fpn = self.fpn(out)

        if self.light_head:
            feature1 = self.light_head1(fpn[0])
            feature2 = self.light_head2(fpn[1])
            feature3 = self.light_head3(fpn[2])
        else:
            feature1 = self.ssh1(fpn[0])
            feature2 = self.ssh2(fpn[1])
            feature3 = self.ssh3(fpn[2])



        yolo_out1 = self.yolo_s8(self.s8_head(feature1))
        yolo_out2 = self.yolo_s16(self.s16_head(feature2))
        yolo_out3 = self.yolo_s32(self.s32_head(feature3))

        yolo_out = [yolo_out1,yolo_out2,yolo_out3]


        if self.training:  # train
            return yolo_out
        elif self.onnx_export :  # export
            return yolo_out  # scores, boxes , landmark: 3780x80, 3780x4
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            return x, p


    def get_yolo_layers(self):
        return [i for i, (n,m) in enumerate(self.named_modules()) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)





import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchstat import stat
import math


# --------------------------------Bias BlueConv Block -----------------------------------#
class addconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(addconv, self).__init__()
        self.in_planes = in_channels
        self.out_planes = out_channels
        self.kernel_size = kernel_size

        self.point_wise = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=True)

        # self.point_wise_bias = nn.Conv2d(in_channels=out_channels,
        #                             #      in_channels=in_channels,
        #                             out_channels=out_channels,
        #                             kernel_size=1,
        #                             stride=1,
        #                             padding=0,
        #                             groups=1,
        #                             bias=True)

        # self.weight =  nn.Parameter(torch.normal(mean=0, std=0.0001, size=(out_channels, out_channels, 1, 1)))
        self.weight = nn.Parameter(torch.normal(mean=0,std=0.0001,size=(out_channels,1,kernel_size,kernel_size)))
        self.weight_2 = nn.Parameter(torch.normal(mean=0,std=0.0001,size=(out_channels,out_channels,1,1)))
        # self.weight = torch.normal(mean=0, std=0.0005, size=(out_channels, 1, kernel_size, kernel_size))
        # self.weight = nn.Parameter(self.weight.repeat(1, out_channels, 1, 1))
        self.depth_wise = nn.Conv2d(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=(kernel_size - 1) // 2,
                                    groups=out_channels,
                                    bias=True)

    def forward(self, x):  #
        # print('x',x.size())
        # print('input',x.size())
        Blue_tmp = self.point_wise(x)  #
        # print('Blue_tmp_point', Blue_tmp.size())
        Blue_tmp = self.depth_wise(Blue_tmp)  #

        bias = F.conv2d(Blue_tmp, self.weight, padding=(self.kernel_size - 1) // 2, groups=self.out_planes)
        bias = F.conv2d(bias,self.weight_2)
        # print('bias:', bias.size())

        out = Blue_tmp+bias


        return out

# --------------------------------Res Block -----------------------------------#
class Res_Block(nn.Module):
    def __init__(self,in_planes):
        super(Res_Block, self).__init__()
        self.conv1=addconv(in_planes,in_planes,3)
        # self.conv1=nn.Conv2d(in_channels=in_planes,out_channels=in_planes,kernel_size=3,padding=1,stride=1)
        self.relu1=nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2=addconv(in_planes,in_planes,3)
        # self.conv2=nn.Conv2d(in_channels=in_planes,out_channels=in_planes,kernel_size=3,padding=1,stride=1)

    def forward(self,x):
        res=self.conv1(x)
        res=self.relu1(res)
        res=self.conv2(res)
        res=self.relu2(res)
        res = torch.add(x, res)
        return res



# --------------------------------Network---------------------------#

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.head_conv=nn.Sequential(

            addconv(9, 20, 3),
            # nn.Conv2d(in_channels=9,out_channels=20,kernel_size=3,padding=1,stride=1),
            # nn.Conv2d(9,32,3,1,1),
            # nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, padding=1, stride=1),
            addconv(20,32,3),
            nn.ReLU(inplace=True)
        )
        #
        self.res_block = nn.Sequential(
            Res_Block(32),
            Res_Block(32),
            Res_Block(32),
            Res_Block(32)

        #
        )

        self.tail_conv=nn.Sequential(
            # nn.Conv2d(32,8,3,1,1),
            addconv(32, 16, 3),
            addconv(16, 8, 3),
            addconv(8, 8, 3)
            # nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,stride=1,padding=1),
            # nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        )

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # def forward(self,pan,lms):
    def forward(self,lms ,pan):
        # print('pan', pan.size())
        # print('lms',lms.size())
        x=torch.cat([pan,lms],1)
        x=self.head_conv(x)
        x = self.res_block(x)
        # x = self.dcm_block(x)
        x=self.tail_conv(x)

        sr=lms+x
        return sr



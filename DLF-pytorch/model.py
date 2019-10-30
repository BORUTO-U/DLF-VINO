import torch
from torch import tensor
from torch import nn
from torch.autograd import Variable
import cv2
import numpy
import torchvision
import os
from torch.nn import functional
from torchvision import models


class Res(nn.Module):
    def __init__(self,ch,size,groups=1,active=nn.Tanh,padding=0):
        super(Res, self).__init__()
        self.ch = ch
        self.pd=((size-1) >> 1) - padding
        self.conv = nn.Sequential(nn.Conv2d(ch, ch, size, groups=groups),
                                  active(),
                                  #nn.Conv2d(ch, ch, 1, 1, (size-1) >> 1, groups=ch, bias=False)
                                  nn.Conv2d(ch, ch, 1, 1, padding, groups=ch, bias=False)
                                  )

    def forward(self, x):
        height=x.size()[2]
        width=x.size()[3]
        pd=self.pd
        return x[:,:,pd:height-pd,pd:width-pd] + self.conv(x)
        #return x + self.conv(x)


class Paddding():
    def __init__(self, channels, paddings):
        if paddings >= 0:
            self.pd=nn.Conv2d(channels, channels, 1, 1, paddings, groups=channels, bias=False)
        else:
            self.pd = nn.ConvTranspose2d(channels, channels, 1, 1, -paddings, groups=channels, bias=False)
        self.pd.train(False)
        self.pd.weight.requires_grad_(False)
        self.pd.weight.zero_()
        self.pd.weight += 1.

    def cuda(self):
        self.pd.cuda()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.pd(x)


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.paddingTranspose = None
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 128, 10, 8),
            nn.Conv2d(128, 128, 1),
            nn.Conv2d(128, 64, 1),
            nn.ConvTranspose2d(64, 1, 10, 8)
        )
        self.paddingTranspose1 = None
        self.conv2 = nn.Sequential(
            nn.Conv2d(2, 64, 5),
            Res(64, 3),
            Res(64, 3),
            nn.Conv2d(64, 128, 3),
            Res(128, 3),
            nn.Conv2d(128, 64, 3),
            Res(64, 3),
            Res(64, 3),
            nn.Conv2d(64, 1, 3),
            Res(1,3)
        )
        self.is_cuda = False

    def cuda(self, device=None):
        nn.Module.cuda(self)
        self.is_cuda = True

    def Padding(self, channels, paddings):
            pd = Paddding(channels, paddings)
            if self.is_cuda:
                pd.cuda()
            return pd

    def forward(self, x):
        if self.paddingTranspose1 is None:
            self.paddingTranspose1 = self.Padding(1, -11)

        y = self.conv1(x)

        y = x[:, 0:1] + y
        ym = self.paddingTranspose1(y[:, 0:1])
        y = torch.cat((y, x[:, 1:2]), 1)

        y = self.conv2(y)

        return y + ym


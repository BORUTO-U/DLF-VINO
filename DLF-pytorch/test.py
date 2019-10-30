import torch
from torch import tensor
from torch import nn
from torch.autograd import Variable
import cv2
import numpy
import torchvision
from torchvision import datasets,transforms
import os
from torch.utils.checkpoint import checkpoint_sequential
from torch.nn import functional
from torchvision import models
from model import Module
from utils import mirror_padding, clip3
import sys

pretrained = sys.argv[1]
#os.environ["CUDA_VISIBLE_DEVICES"] ='3'
gpu=[0]

dire= sys.argv[2] 
width = int(sys.argv[3])
height = int(sys.argv[4]) 

class SrDataset(torch.utils.data.Dataset):
    def __init__(self, phase, dire, width, height):
        super(SrDataset,self).__init__()
        idx = 0
        if phase == 'valid':
            size = 192
            iv = range(1, 193)
        self.x = torch.zeros(size, 2, height, width)
        self.t = torch.zeros(size, 1, height, width)
        s = torch.zeros(size, 1, height, width)
        for i in iv:
            self.t[idx]    = tensor(cv2.imread(dire+     'org/{:d}.bmp'.format(i))).transpose(0, 2).transpose(1, 2)[0:1]
            self.x[idx][0] = tensor(cv2.imread(dire+'unfilter/{:d}.bmp'.format(i))).transpose(0, 2).transpose(1, 2)[0:1]
            self.x[idx][1] = tensor(cv2.imread(dire+    'pred/{:d}.bmp'.format(i))).transpose(0, 2).transpose(1, 2)[0:1]
            s[idx] = tensor(cv2.imread(dire+     'sao/{:d}.bmp'.format(i))).transpose(0, 2).transpose(1, 2)[0:1]
            # self.x[idx][1] = self.x[idx][0] - self.x[idx][1]
        
            idx += 1

        mse = (self.x[:,0:1]-self.t).pow(2).mean()
        psnr = 10*((255*255/mse).log() / tensor(10.).log())
        print('{} input psnr: {:.4f}\n'.format(phase, psnr))
        mse = (s-self.t).pow(2).mean()
        psnr = 10*((255*255/mse).log() / tensor(10.).log())
        print('{} sao psnr: {:.4f}\n'.format(phase, psnr))
        #self.t /= 256
        self.x /= 256
        #self.x = self.x[:, :, 3:(height-3), 3:(width-3)]
        #self.t = self.t[:, :, 3:(height-3), 3:(width-3)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.t[idx], idx)


# ===================================================================================================== #
batch_size = {'train':32, 'valid':32}
dataloader = {phase: torch.utils.data.DataLoader(dataset=SrDataset(phase,dire,width,height), batch_size=batch_size[phase], shuffle=False)
              for phase in ['valid']
              }

use_gpu = torch.cuda.is_available()
module = Module()
module.load_state_dict(torch.load(pretrained))

fid = open('parameters','wb+')
for param in module.parameters():
    b = param.data.numpy()    
    fid.write(b)
fid.close()

if use_gpu:
    module.cuda()
    module = nn.DataParallel(module, gpu)

for stage in ([0]*1):
#   for epoch in range(1):
        for phase in ["valid"]:
            print("Testing...")
            module.train(False)
            for param in module.parameters():
                param.requires_grad_(False)

            running_dist = 0.
            for batch, data in enumerate(dataloader[phase], 1):
                x, t, idx = data
                if use_gpu:
                    x = x.cuda()
                    t = t.cuda()
                batch_size = 32
                bs = 4
                                
                for i in range(0, batch_size, bs):
                    #xm = x[i:i + bs, :, 3:(height-3), 3:(width-3)]
                    xm = mirror_padding(x[i:i+bs], 21, True)
                    y0 = module(xm)
                    if i == 0:
                        y = y0
                    else:
                        y = torch.cat((y, y0))
                
                #x[:, 0:1, 8:(height-8), 8:(width-8)] = y
                #y = x[:, 0:1]
                y = y[:, : ,10:(height+10), 10:(width+10)]
                #t = t[:, :, 8:(height-8), 8:(width-8)]

                y8 = clip3((y.data*256).round(),0,255)
                x8 = clip3((x.data[:,0:1]*256).round(),0,255)
                
                dist = (y8 - t.data).pow(2).sum()
                running_dist += dist
                
                for i in range(batch_size):
                    print(
                      -( 10*((255*255/( (x8[i] - t.data[i]).pow(2).sum()/t.data[i].numel() )).log() / tensor(10.).log().cuda()) )
                      +( 10*((255*255/( (y8[i] - t.data[i]).pow(2).sum()/t.data[i].numel() )).log() / tensor(10.).log().cuda()) )
                      )

                np_y = y8.transpose(1, 3).transpose(1, 2).cpu().numpy()
                for i, image in enumerate(np_y):
                    cv2.imwrite('./result/{}/{:d}.bmp'.format('test', idx[i]), image)

            mse = running_dist / (y8.numel() * batch)
            psnr = 10*((255*255/mse).log() / tensor(10.).log().cuda())
            print('running psnr: {:.4f}\n'.format(psnr))


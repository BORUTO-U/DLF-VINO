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
# import model
from model import Module
import sys

pretrained = None
#os.environ["CUDA_VISIBLE_DEVICES"] ='2'
gpu=[0]

img_dir = sys.argv[1]
width = int(sys.argv[2])
height = int(sys.argv[3]) 

class SrDataset(torch.utils.data.Dataset):
    def __init__(self, phase, width, height, img_dir):
        super(SrDataset,self).__init__()
        idx = 0
        if phase == 'valid':
            size = 20
            iv = range(480, 500)
        else:
            size = 480
            iv = range(0, 480)
        self.x = torch.zeros(size, 2, height, width)
        self.t = torch.zeros(size, 1, height, width)

        for i in iv:
            self.t[idx] = tensor(cv2.imread(img_dir+'org/{:d}.bmp'.format(i))
                                     ).transpose(0, 2).transpose(1, 2)[0:1]
            self.x[idx][0] = tensor(cv2.imread(img_dir+'unfilter/{:d}.bmp'.format(i))
                                     ).transpose(0, 2).transpose(1, 2)[0:1]
            self.x[idx][1] = tensor(cv2.imread(img_dir+'pred/{:d}.bmp'.format(i))
                                     ).transpose(0, 2).transpose(1, 2)[0:1]
            idx += 1

        mse = (self.x[:, 0:1]-self.t[:, 0:1]).pow(2).mean()
        psnr = 10*((255*255/mse).log() / tensor(10.).log())
        print('{} input psnr: {:.4f}\n'.format(phase, psnr))
        self.t /= 256
        self.x /= 256
        self.t = self.t[:, :, 14:(height-14), 14:(width-14)]
        self.x = self.x[:, :, 3:(height-3), 3:(width-3)]
        # self.x[:, 1] = self.x[:, 0] - self.x[:, 1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.t[idx], idx)


batch_sz = {'train':12, 'valid':10}
dataloader = {phase: torch.utils.data.DataLoader(dataset=SrDataset(phase, width, height, img_dir), batch_size=batch_sz[phase], shuffle=True)
              for phase in ['train', 'valid']
              }

use_gpu = torch.cuda.is_available()

module = Module()

if pretrained is None:
    for param in module.parameters():
        print(param.size())
        param.data.normal_(0.001, 0.05)
else:
    module.load_state_dict(torch.load(pretrained))

if use_gpu:
    module.cuda()
    #module = nn.DataParallel(module, gpu)

# print(module)
loss=nn.MSELoss()
optimizer = torch.optim.Adam(module.parameters(), lr=1)
#optimizer = nn.DataParallel(optimizer,gpu).module

''''''
lam = torch.tensor(0.025).cuda()
lam = torch.autograd.variable(lam, requires_grad=True)
optimizer_lam = torch.optim.Adam([lam],lr=3e-4)
lam.backward()
optimizer.zero_grad()
lam.requires_grad_(False)
''''''
height -= 28
width -= 28

iterations = 0
for stage in ([0]):
    if stage == 0:
        rg = range(0,30000) #700
        lr = 0.0003 #* pow(0.9997,2100)
    else:
        rg = range(2)
        lr = 0.0003
    for epoch in rg:
        lr *= 0.9997
        print("\nEpoch {:d}".format(epoch))
        for phase in ["train","valid"]:

            if phase == "train":
                print("Training...")
                module.train(True)
                for param in module.parameters():
                    param.requires_grad_(True)
                for param_g in optimizer.param_groups:
                    param_g['lr']=lr
            else:
                print("Validing...")
                module.train(False)
                for param in module.parameters():
                    param.requires_grad_(False)

            running_dist = 0.
            for batch, data in enumerate(dataloader[phase], 1):
                x, t, idx = data
                if use_gpu:
                    x = x.cuda()
                    t = t.cuda()
                # print('\ninput dist:  {:.4f}'.format((x - t).pow(2).mean()))
                batch_size = batch_sz[phase]
                if phase == 'train':
                        optimizer.zero_grad()
                        ''''''
                        y0 = module(x)
                        # height = y0.size()[2]
                        # width = y0.size()[3]
                        gu = y0[:, :, 0:height-1, 0:width-1] - y0[:, :, 0:height-1, 1:width]
                        gv = y0[:, :, 0:height-1, 0:width-1] - y0[:, :, 1:height, 0:width-1]
                        guv = torch.cat((gu, gv),1)
                        a_var = guv.abs().mean()
                        '''
                        (a_var.exp() * lam).backward()
                        gr = []
                        for param in module.parameters():
                            if param.grad.data.abs().mean() < (0.05/lr):
                                gr += [param.grad.data / lam]
                            else:
                                print('fly!')
                                gr = None
                                optimizer.zero_grad()
                                break
                        ''''''
                        y0 = module(x)
                        '''
                        L = (loss(y0, t) + (a_var * lam)).exp()
                        L.backward()

                        y = y0.data

                        optimizer.step()
                        '''
                        for batchv, data in enumerate(dataloader['valid'], 1):
                            optimizer.zero_grad()
                            x, tv, _ = data
                            if use_gpu:
                                x = x.cuda()
                                tv = tv.cuda()
                            batch_size = batch_sz['valid']
                            y0 = module(x)
                            loss(y0, tv).backward()
                            optimizer_lam.zero_grad()
                            if gr is None:
                                break
                            else:
                                for pa_idx,param in enumerate(module.parameters(),0):
                                    lam.grad -= (param.grad.data*gr[pa_idx]).sum()
                            optimizer_lam.step()
                            if lam.abs() < 0.5:
                                break
                            else:
                                lam[:] = 0.02
                                break
                        '''
                else:
                    y = module(x)

                dist = ((y.data*256).floor() - (t.data*256).floor()).pow(2.).sum()
                running_dist += dist

                # print('batch{:3d} dist: {:.4f}'.format(batch, dist))
                # print('\n')
                # for i in range(0,8):
                #     print((y.data[i]-t.data[i]).pow(2).mean())

                if epoch % 15 == 0:
                    np_y = (y*256).data.transpose(1, 3).transpose(1, 2).squeeze().cpu().numpy()
                    for i, image in enumerate(np_y):
                        cv2.imwrite('./result/{}/{:d}.bmp'.format(phase, idx[i]), image)
            mse = running_dist / (y.data.cpu().numpy().size*batch)
            psnr = 10*((255*255/mse).log() / tensor(10.).log().cuda())
            print('running psnr: {:.4f}\n'.format(psnr))
            print('lambda:{}'.format(lam))
        torch.save(module.state_dict(), './autosave/module.pth')
        torch.save(optimizer.state_dict(),'./autosave/optimizer.pth')
        torch.save(optimizer_lam.state_dict(), './autosave/optimizer_lam.pth')
        if epoch%50 == 0:
            torch.save(module.state_dict(), './pth/module_iter_{:d}.pth'.format(iterations))
            print('saved - pth/module_iter_{:d}.pth'.format(iterations))
        iterations += 1
torch.save(module.state_dict(), 'module.pth')

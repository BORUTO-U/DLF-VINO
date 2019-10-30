import torch

def mirror_padding(x, pd=5, is_cuda=True):
    height = x.size()[2]
    width = x.size()[3]
    batch_size = x.size()[0]
    ch = x.size()[1]
    xm = torch.zeros(batch_size, ch, height+2*pd, width+2*pd)
    if is_cuda:
        xm = xm.cuda()
    xm[:, :, pd:(height+pd), pd:(width+pd)] = x
    xm[:, :, range(pd-1,-1,-1), :] = xm[:, :, range(pd,2*pd), :]
    xm[:, :, range(height+pd,height+2*pd), :] = xm[:, :, range(height+pd-1,height-1,-1), :]
    xm[:, :, :, range(pd-1,-1,-1)] = xm[:, :, :, range(pd,2*pd)]
    xm[:, :, :, range(width+pd,width+2*pd)] = xm[:, :, :, range(width+pd-1,width-1,-1)]
    return xm

def clip3(x,a,b):
    relu=torch.nn.ReLU()
    x=relu(x-a)+a
    x=-relu(b-x)+b
    return x

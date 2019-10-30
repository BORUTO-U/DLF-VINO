import tensorflow as tf
import numpy as np
global m_cnt
from module import Module
import torch
import torchvision
from torchvision import datasets,transforms
import cv2
import os

#os.environ["CUDA_VISIBLE_DEVICES"] ='2,3'

width = 176
height = 144
qp = '23'
img_dir = 'HM_Datasets/ai/qp'+qp+'/Foreman_320x256_ai_qp'+qp+'/'


class SrDataset(torch.utils.data.Dataset):
    def __init__(self, phase, width, height, img_dir):
        super(SrDataset,self).__init__()
        idx = 0
        if phase == 'valid':
            size = 100
            iv = range(0, 100)
        else:
            size = 480
            iv = range(0, 480)
        self.x = torch.zeros(size, height, width, 2)
        self.t = torch.zeros(size, height, width, 1)

        for i in iv:
            self.t[idx] = torch.tensor(cv2.imread(img_dir+'org/{:d}.bmp'.format(i)))[0:height,0:width,0:1]
            self.x[idx][:,:,0:1] = torch.tensor(cv2.imread(img_dir+'unfilter/{:d}.bmp'.format(i)))[0:height,0:width,0:1]
            self.x[idx][:,:,1:2] = torch.tensor(cv2.imread(img_dir+'pred/{:d}.bmp'.format(i)))[0:height,0:width,0:1]
            idx += 1

        mse = (self.x[:, :, :, 0:1]-self.t[:, :, :, 0:1]).pow(2.).mean()
        psnr = 10*((255*255/mse).log() / torch.tensor(10.).log())
        print('{} input psnr: {:.4f}\n'.format(phase, psnr))
        self.t /= 256
        self.x /= 256
        self.x = self.x[:, 3:(height-3), 3:(width-3), :]
        self.t = self.t[:, 14:(height-14), 14:(width-14), :]
        # self.x[:, 1] = self.x[:, 0] - self.x[:, 1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.t[idx], idx)


batch_sz = {'train':1, 'valid':1}
dataloader = {phase: torch.utils.data.DataLoader(dataset=SrDataset(phase, width, height, img_dir), batch_size=batch_sz[phase], shuffle=True)
              for phase in ['valid']
              }

model = Module()
input_v0 = tf.placeholder(tf.float32, [1, height-6,width-6, 1])
input_v1 = tf.placeholder(tf.float32, [1, height-6,width-6, 1])
output_v = model(input_v0,input_v1)
#output_v = output_v[:, 6:-6, 6:-6, :]

fid1=open('parameters_ai_qp'+qp+'.pa', 'rb')
#print(model.parameters())
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    #constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, [output_v.name])
    for param in model.parameters():
         shape = param.shape.as_list()
         if shape.__len__() == 4:
             val = torch.zeros(shape[3],shape[2],shape[0],shape[1]).numpy()
             fid1.readinto(val)
             print(val.shape)
             val = val.transpose((2, 3, 1, 0))
         elif shape.__len__() == 1:
             val = torch.zeros(shape[0]).numpy()
             print(val.shape)
             fid1.readinto(val)
         else:
             val = []
         sess.run(param.assign(val))
    #constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, [output_v.name])

    for epoch in range(1):
        print('epoch: {:d}'.format(epoch))
        print('validing')
        running_dist = 0.
        for batch, data in enumerate(dataloader['valid'], 1):
            x, t, idx = data
            x = x.numpy()
            t = t.numpy()
            y = sess.run((output_v), feed_dict={input_v0: x[:,:,:,0:1], input_v1: x[:,:,:,1:2]})

            y = y[:,11:-11,11:-11,:]
            dist = np.float_power(np.floor(y * 256) - np.floor(t * 256), 2.).sum()
            running_dist += dist

            if batch == 1:
                for i, image in enumerate(y):
                    cv2.imwrite('./result/{}/{:d}.bmp'.format('valid', idx[i]), image * 256)

        mse = running_dist / (y.size * batch)
        psnr = 10 * np.log(255 * 255 / mse) / np.log(10.)
        print('running psnr: {:.4f}\n'.format(psnr))
        if epoch % 50 == 0:
            fid = open('./pa/epoch_{:d}.pa'.format(epoch),'wb+')
            for param in model.parameters():
                pa = sess.run(param)
                fid.write(pa)
    graph_def=tf.get_default_graph().as_graph_def()
    constant_graph = tf.graph_util.convert_variables_to_constants(sess, graph_def,[output_v.op.name])
    print(input_v0.name)
    print(input_v1.name)
    print(output_v.op.name)
    print(output_v.name)
    with tf.gfile.FastGFile('./DLF'+qp+'.pb', mode='wb+') as f:
        f.write(constant_graph.SerializeToString())

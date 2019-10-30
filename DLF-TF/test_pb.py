import tensorflow as tf
from tensorflow.python.platform import gfile
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
qp='23'
img_dir = 'HM_Datasets/ai/qp23/Foreman_320x256_ai_qp23/'


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


batch_sz = {'valid':1}
dataloader = {phase: torch.utils.data.DataLoader(dataset=SrDataset(phase, width, height, img_dir), batch_size=batch_sz[phase], shuffle=True)
              for phase in ['valid']
              }

with tf.Session() as sess:

    with gfile.FastGFile( 'DLF'+qp+'.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图

    # 需要有一个初始化的过程
    sess.run(tf.global_variables_initializer())

    # 需要先复原变量
    #print(sess.run('b:0'))

    # 输入
    input_v0 = sess.graph.get_tensor_by_name('Placeholder:0')
    input_v1 = sess.graph.get_tensor_by_name('Placeholder_1:0')
    # op
    output_v = sess.graph.get_tensor_by_name('add_13:0')

    for epoch in range(1):
        print('epoch: {:d}'.format(epoch))
        print('validing')
        running_dist = 0.
        for batch, data in enumerate(dataloader['valid'], 1):
            x, t, idx = data
            x = x.numpy()
            t = t.numpy()
            y = sess.run((output_v), feed_dict={input_v0: x[:,:,:,0:1],input_v1: x[:,:,:,1:2]})
            y = y[:, 11:-11, 11:-11, :]
            dist = np.float_power(np.floor(y * 256) - np.floor(t * 256), 2.).sum()
            running_dist += dist

            for i, image in enumerate(y):
                cv2.imwrite('./result/{}/{:d}.bmp'.format('valid', idx[i]), image * 256)

        mse = running_dist / (y.size * batch)
        psnr = 10 * np.log(255 * 255 / mse) / np.log(10.)
        print('running psnr: {:.4f}\n'.format(psnr))

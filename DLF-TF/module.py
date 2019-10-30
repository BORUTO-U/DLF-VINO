import tensorflow as tf
import numpy as np
import nn


class Res(nn.Module):
    def __init__(self, ch, size,name=None):
        super(Res, self).__init__()
        self.conv = nn.Conv2d(ch, ch, size, padding='SAME',name=name)
        if name is None:
            name = None
        else:
            name = name+'_scale'
        self.scale = tf.Variable(tf.random_normal(shape=[ch], mean=0., stddev=0.5), name=name)
        self.register_all_modules()
        self.parameters_ += [self.scale]

    def __call__(self, x):
        y = self.conv(x)
        y = tf.nn.tanh(y)
        y = y*self.scale
        return x + y


class Module(nn.Module):
  def __init__(self):
    super(Module, self).__init__()
    self.conv0 = nn.Sequential([
        nn.Conv2d(2, 128, 10, 4),
        nn.DownSample2d(in_channels=128),
        nn.Conv2d(128, 128, 1),
        nn.Conv2d(128, 64, 1),
        nn.UpSample2d(in_channels=64, stride=4),
        nn.UpSample2d(in_channels=64),
        nn.ConvTranspose2d( 64, 1, 10)
    ])
    self.register_module(self.conv0)
    self.conv1 = nn.Sequential([
        nn.Conv2d(2, 64, 5,padding='SAME'),
        Res(64, 3),
        Res(64, 3),
        nn.Conv2d(64, 128, 3,padding='SAME'),
        Res(128, 3),
        nn.Conv2d(128, 64, 3,padding='SAME'),
        Res(64, 3),
        Res(64, 3),
        nn.Conv2d(64, 1, 3,padding='SAME'),
        Res(1, 3)
    ])
    self.register_module(self.conv1)
    #self.register_all_modules()

  def __call__(self, input0,input1):
      output = tf.concat((input0, input1),3)
      output = self.conv0(output)
      output = input0 + output
      output1 = tf.concat((output,input1), 3)
      output1 = self.conv1(output1)
      return output1 + output

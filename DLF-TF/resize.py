from nn import Module
import numpy as np
import tensorflow as tf

class DownSample2d(Module):
    def __init__(self, in_channels, out_channels=None, kernal_size = 1, stride = 2, padding='VALID', groups=1, name=None):
        super(DownSample2d, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if name is None:
            namebias = None
            nameweight = None
        else:
            nameweight = name + '_weight'
            namebias = name + '_bias'
        if np.asarray(kernal_size).size == 1:
            weight_size = [kernal_size,kernal_size,in_channels,int(out_channels/groups)]
        else:
            weight_size = [kernal_size[0], kernal_size[1], in_channels, int(out_channels/groups)]

        mask=np.zeros(weight_size)
        for c in range(in_channels):
            mask[0, 0, c, c] = 1.
        self.weight = tf.constant(value=mask, name=nameweight,dtype=float)
        #self.bias = tf.Variable(tf.random_normal([out_channels], 0., 0.1), name=namebias)

        if (padding == 0)or(padding == 'VALID'):
            self.padding = 'VALID'
        else:
            self.padding = 'SAME'
        if np.asarray(stride).size == 1:
            self.stride = [1,stride, stride,1]
        else:
            self.stride = [1, stride[0], stride[1], 1]
        self.groups = groups
        self.in_channels = in_channels
        self.parameters_ = []

    def __call__(self,input):
        gsize = int(self.in_channels / self.groups)
        for gstart in range(0, self.in_channels, gsize):
            if self.groups == 1:
                value = input
            else:
                value = input[:,:,:,gstart:gstart+gsize]
            conv = tf.nn.conv2d(input=value,
                                filter=(self.weight[:, :, gstart:gstart+gsize, :]),
                                strides=self.stride,
                                padding=self.padding,
                                )
            if gstart == 0:
                output = conv
            else:
                output = tf.concat((output,conv), 3)
        #bias=tf.nn.bias_add(output,self.bias)
        return output


class UpSample2d(Module):
    def __init__(self, in_channels, out_channels=None, kernal_size = 1, stride = 2, padding='VALID', groups=1, name=None):
        super(UpSample2d, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if name is None:
            namebias = None
            nameweight = None
        else:
            nameweight=name+'_weight'
            namebias=name+'_bias'
        if np.asarray(kernal_size).size == 1:
            weight_size = [kernal_size,kernal_size, int(out_channels/groups), in_channels]
        else:
            weight_size = [kernal_size[0], kernal_size[1], int(out_channels/groups), in_channels]
        self.out_channels=out_channels
        self.weight_size=weight_size

        mask = np.zeros(weight_size)
        for c in range(in_channels):
            mask[0, 0, c, c] = 1.
        self.weight = tf.constant(value=mask, name=nameweight,dtype=float)
        #self.bias = tf.Variable(tf.random_normal([out_channels], 0., 0.1), name=namebias)

        if (padding == 0)or(padding == 'VALID'):
            self.padding = 'VALID'
        else:
            self.padding = 'SAME'
        if np.asarray(stride).size == 1:
            self.stride = [1,stride, stride,1]
        else:
            self.stride = [1, stride[0], stride[1], 1]
        self.groups = groups
        self.in_channels = in_channels
        self.parameters_ = []

    def __call__(self,input):
        input_shape = tf.shape(input)
        output_shape = [
            input_shape[0],
            (input_shape[1]-1)*self.stride[1]+self.weight_size[0],
            (input_shape[2]-1)*self.stride[2]+self.weight_size[1],
            int(self.out_channels/self.groups)
        ]
        gsize = int(self.in_channels / self.groups)
        for gstart in range(0, self.in_channels, gsize):
            if self.groups == 1:
                value = input
            else:
                value = input[:,:,:,gstart:gstart+gsize]
            conv = tf.nn.conv2d_transpose(
                value=value,
                filter=(self.weight[:, :, :, gstart:gstart+gsize]),
                output_shape=output_shape,
                strides=self.stride,
                padding=self.padding
            )
            if gstart == 0:
                output = conv
            else:
                output = tf.concat((output,conv), 3)

        #bias=tf.nn.bias_add(conv,self.bias)
        return output

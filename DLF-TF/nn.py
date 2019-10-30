import tensorflow as tf
import numpy as np


class Module(object):
    def __init__(self):
        self.parameters_ = []
        self.modules_ = [self]

    def register_module(self, module):
        self.parameters_ += module.parameters()
        self.modules_ += module.modules()
        return module

    def register_all_modules(self):
        for name, value in vars(self).items():
            if hasattr(value, "modules") and hasattr(value, "parameters"):
                if callable(getattr(value, "modules")) and callable(getattr(value, "parameters")):
                    self.register_module(value)

    def parameters(self):
        return self.parameters_

    def modules(self):
        return self.modules_
    
    
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride=1, padding='VALID', groups=1, name=None):
        super(Conv2d, self).__init__()
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

        self.weight = tf.Variable(
            tf.random_normal(
                shape=weight_size,
                mean=0.,
                stddev=1./float(weight_size[0]*weight_size[1]*weight_size[2])
            ),
            name=nameweight
        )
        self.bias = tf.Variable(tf.random_normal([out_channels], 0., 0.1), name=namebias)
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
        self.parameters_ = [self.weight, self.bias]

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
                                padding=self.padding
                                )
            if gstart == 0:
                output = conv
            else:
                output = tf.concat((output,conv), 3)
        bias=tf.nn.bias_add(output,self.bias)
        return bias


class ConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride=1, padding='VALID', groups=1, name=None):
        super(ConvTranspose2d, self).__init__()
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
        self.weight=tf.Variable(tf.random_normal(weight_size, 0., 1./float(in_channels)), name=nameweight)
        self.bias = tf.Variable(tf.random_normal([out_channels], 0., 0.1), name=namebias)
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
        self.parameters_ = [self.weight, self.bias]

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

        bias=tf.nn.bias_add(conv,self.bias)
        return bias


class Sequential(Module):
    def __init__(self, modules):
        super(Sequential, self).__init__()
        self.sequential = []
        for layer in modules:
            self.sequential += [self.register_module(layer)]

    def __call__(self, input):
        output = input
        for layer in self.sequential:
            output = layer(output)
        return output


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def __call__(self, input):
        return tf.nn.tanh(input)


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def __call__(self, input):
        return tf.nn.sigmoid(input)


from resize import DownSample2d, UpSample2d

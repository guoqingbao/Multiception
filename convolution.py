'''
Author: Guoqing Bao
School of Computer Science, The University of Sydney
04/09/2020

Citation:
Guoqing Bao, Manuel B. Graeber, Xiuying Wang, "Depthwise Multiception Convolution for Reducing Network Parameters without Sacrificing Accuracy," 
2020 16th International Conference on Control, Automation, Robotics and Vision (ICARCV), 2020, pp. 747-752, doi: 10.1109/ICARCV50220.2020.9305369.

'''

'''
Implementation of four different types of convolutions
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

#standard convolution (used in baseline models)
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv3x3_bn(in_planes, out_planes, stride=1, kernels=[]):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
                                                nn.BatchNorm2d(out_planes))
#depthwise separable convolution
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, stride, kernels=[]):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, stride=stride, padding=1, groups=nin, bias=False)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        self.bn = nn.BatchNorm2d(nout)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn(self.pointwise(out))
        return out

#multiception convolution
class Multiception(nn.Module):
    def __init__(self, in_channel, out_channel, stride, kernels):
        super(Multiception, self).__init__()
        padding_dict = {1:0, 3:1, 5:2, 7:3}
        self.seps = nn.ModuleList()
        for kernel in kernels:
            sep = nn.Conv2d(in_channel,in_channel, kernel_size = kernel,stride =1,padding = padding_dict[kernel],dilation=1,groups=in_channel, bias=False)
            self.seps.append(sep)
        self.bn1 = nn.BatchNorm2d(in_channel*len(kernels)) 
        self.pointwise = nn.Conv2d(in_channel*len(kernels), out_channel, 1, stride, 0, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channel)       

    def forward(self, x):
        seps = []
        for sep in self.seps:
            seps.append(sep(x))
        out_seq = torch.cat(seps, dim=1)
        out = self.pointwise(self.bn1(out_seq))
        out = self.bn2(out)
        return out 

#MixedConv
def _split_channels(total_filters, num_groups):
    """
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py#L33
    """
    split = [total_filters // num_groups for _ in range(num_groups)]
    split[0] += total_filters - sum(split)
    return split

class MixConv(nn.Module):
    """
    Mixed convolution layer from 'MixConv: Mixed Depthwise Convolutional Kernels,' https://arxiv.org/abs/1907.09595.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of int, or tuple/list of tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of int, or tuple/list of tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    axis : int, default 1
        The axis on which to concatenate the outputs.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 kernels = [],
                 kernel_size = [3, 5, 7],
                 dilation=1,
                 groups=1,
                 bias=False,
                 axis=1):
        super(MixConv, self).__init__()
        padding_dict = {1:0, 3:1, 5:2, 7:3}
        kernel_count = len(kernel_size)
        self.splitted_in_channels = self.split_channels(in_channels, kernel_count)
        splitted_out_channels = self.split_channels(out_channels, kernel_count)
        for i, ksize in enumerate(kernel_size):
            self.add_module(
                name=str(i),
                module=nn.Conv2d(
                    in_channels=self.splitted_in_channels[i],
                    out_channels=splitted_out_channels[i],
                    kernel_size=ksize,
                    stride=stride,
                    padding=padding_dict[ksize],
                    dilation=dilation,
                    groups=(splitted_out_channels[i] if out_channels == groups else groups),
                    bias=bias))
        self.axis = axis

    def forward(self, x):
        xx = torch.split(x, self.splitted_in_channels, dim=self.axis)
        out = [conv_i(x_i) for x_i, conv_i in zip(xx, self._modules.values())]
        x = torch.cat(tuple(out), dim=self.axis)
        return x

    @staticmethod
    def split_channels(channels, kernel_count):
        splitted_channels = [channels // kernel_count] * kernel_count
        splitted_channels[0] += channels - sum(splitted_channels)
        return splitted_channels

'''
Author: Guoqing Bao
School of Computer Science, The University of Sydney
04/09/2020

Reference:
Guoqing Bao, Manuel B. Graeber, Xiuying Wang, "Depthwise Multiception Convolution for Reducing Network Parameters without Sacrificing Accuracy", 
16th International Conference on Control, Automation, Robotics and Vision (ICARCV 2020), In Press.

'''

'''Shake-Shake regularization in PyTorch. Modification to accept Multiception convolution

Reference:
Xavier Gastaldi. Shake-Shake regularization of 3-branch residual networks. 
International Conference on Learning Representation (Workshop) 2017
'''


import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from shakeshake import ShakeShake
from shakeshake import Shortcut
from convolution import conv3x3, conv3x3_bn


class ShakeBlock(nn.Module):

    def __init__(self, convBN, in_ch, out_ch, stride, kernels):
        super(ShakeBlock, self).__init__()
        self.equal_io = in_ch == out_ch
        self.shortcut = self.equal_io and None or Shortcut(in_ch, out_ch, stride=stride)

        self.branch1 =  self._make_branch(convBN, in_ch, out_ch, stride, kernels) 
        self.branch2 =  self._make_branch(convBN, in_ch, out_ch, stride, kernels) 

    def forward(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        h = ShakeShake.apply(h1, h2, self.training)
        h0 = x if self.equal_io else self.shortcut(x)
        return h + h0

    def _make_branch(self, convBN, in_ch, out_ch, stride, kernels):
        return nn.Sequential(
            nn.ReLU(inplace=False),
            convBN(in_ch, out_ch, stride, kernels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch))


class ShakeResNet(nn.Module):

    def __init__(self, convBN, depth, w_base, label, cifar=True):
        super(ShakeResNet, self).__init__()
        n_units = (depth - 2) / 6

        in_chs = [16, w_base, w_base * 2, w_base * 4]
        self.in_chs = in_chs

        self.c_in = nn.Conv2d(3, in_chs[0], 3, padding=1)
        self.layer1 = self._make_layer(convBN, n_units, in_chs[0], in_chs[1], 1 if cifar else 2, [3,5,7])
        self.layer2 = self._make_layer(convBN, n_units, in_chs[1], in_chs[2], 2, [3,5])
        self.layer3 = self._make_layer(convBN, n_units, in_chs[2], in_chs[3], 2, [3,3])
        self.fc_out = nn.Linear(in_chs[3], label)
        self.cifar = cifar
        # Initialize paramters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        h = self.c_in(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = F.relu(h)
        if self.cifar:
            h = F.avg_pool2d(h, 8)
            h = h.view(-1, self.in_chs[3])
        else:
            h = F.adaptive_avg_pool2d(h, 1)
            h = h.view(h.size(0), -1)

        h = self.fc_out(h)
        return h

    def _make_layer(self, convBN, n_units, in_ch, out_ch, stride, kernels):
        layers = []
        for i in range(int(n_units)):
            layers.append(ShakeBlock(convBN, in_ch, out_ch, stride, kernels))
            in_ch, stride = out_ch, 1
        return nn.Sequential(*layers)
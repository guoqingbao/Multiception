'''
Author: Guoqing Bao
School of Computer Science, The University of Sydney
04/09/2020

Reference:
Guoqing Bao, Manuel B. Graeber, Xiuying Wang, "Depthwise Multiception Convolution for Reducing Network Parameters without Sacrificing Accuracy", 
16th International Conference on Control, Automation, Robotics and Vision (ICARCV 2020), In Press.

'''

'''Squeeze-and-Excitation Networks in PyTorch. Modification to accept Multiception convolution

Reference:
Hu J, Shen L, Sun G. Squeeze-and-excitation networks. 
In Proceedings of the IEEE conference on computer vision and pattern recognition 2018 (pp. 7132-7141).

'''

import torch.nn as nn
from torchvision.models import ResNet
import torch
import torch.nn.functional as F

from convolution import conv3x3, conv3x3_bn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CifarSEBasicBlock(nn.Module):
    def __init__(self, convBN, inplanes, planes, stride=1, reduction=16, kernels=[3,3]):
        super(CifarSEBasicBlock, self).__init__()
        self.conv_bn = convBN(inplanes, planes, stride, kernels)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv_bn(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class CifarSEResNet(nn.Module):
    def __init__(self, block, convBN, n_size, cifar, num_classes=10, reduction=16):
        super(CifarSEResNet, self).__init__()
        self.inplane = 16
        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=3, stride=1 if cifar else 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, convBN, 16, blocks=n_size, stride=1, reduction=reduction, kernels=[3, 5, 7])
        self.layer2 = self._make_layer(
            block, convBN, 32, blocks=n_size, stride=2, reduction=reduction, kernels=[3, 5])
        self.layer3 = self._make_layer(
            block, convBN, 64, blocks=n_size, stride=2, reduction=reduction, kernels=[3, 3])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, convBN, planes, blocks, stride, reduction, kernels):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(convBN, self.inplane, planes, stride, reduction, kernels))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def se_resnet20(convBN=conv3x3_bn, cifar=True, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model = CifarSEResNet(CifarSEBasicBlock, convBN, 3, cifar, **kwargs)
    return model


def se_resnet32(convBN=conv3x3_bn, cifar=True, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEResNet(CifarSEBasicBlock, convBN, 5, cifar, **kwargs)
    return model


def se_resnet56(convBN=conv3x3_bn, cifar=True, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEResNet(CifarSEBasicBlock, convBN, 9, cifar, **kwargs)
    return model

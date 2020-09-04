'''
Author: Guoqing Bao
School of Computer Science, The University of Sydney
04/09/2020

Reference:
Guoqing Bao, Manuel B. Graeber, Xiuying Wang, "Depthwise Multiception Convolution for Reducing Network Parameters without Sacrificing Accuracy", 
16th International Conference on Control, Automation, Robotics and Vision (ICARCV 2020), In Press.

'''

'''ResNet in PyTorch. Modification to accept Multiception convolution
Reference:
He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. 
InProceedings of the IEEE conference on computer vision and pattern recognition 2016 (pp. 770-778).
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from convolution import conv3x3, conv3x3_bn


class BasicBlock(nn.Module):
    def __init__(self, convBN, inplanes, planes, stride=1, kernels=[3,3]):
        super(BasicBlock, self).__init__()

        self.conv_bn = convBN(inplanes, planes, stride, kernels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, convBN, n_size, num_classes=10, cifar=True):
        super(ResNet, self).__init__()
        self.inplane = 16
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, convBN, 16, blocks=n_size, stride=1, kernels=[3, 5, 7])
        self.layer2 = self._make_layer(block, convBN, 32, blocks=n_size, stride=2, kernels=[3, 5])
        self.layer3 = self._make_layer(block, convBN, 64, blocks=n_size, stride=2, kernels=[3, 3])
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

    def _make_layer(self, block, convBN, planes, blocks, stride, kernels):

        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(convBN, self.inplane, planes, stride, kernels))
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


def resnet20(num_classes=10, convBN=conv3x3_bn, cifar=True, **kwargs):
    model = ResNet(BasicBlock, convBN, 3, num_classes, cifar, **kwargs)
    return model


def resnet32(num_classes=10, convBN=conv3x3_bn, cifar=True, **kwargs):
    model = ResNet(BasicBlock, convBN, 5, num_classes, cifar, **kwargs)
    return model

def resnet50(num_classes=10, convBN=conv3x3_bn, cifar=True, **kwargs):
    model = ResNet(BasicBlock, convBN, 8, num_classes, cifar, **kwargs)
    return model

def resnet56(num_classes=10, convBN=conv3x3_bn, cifar=True, **kwargs):
    model = ResNet(BasicBlock, convBN, 9, num_classes, cifar, **kwargs)
    return model


def resnet110(num_classes=10, convBN=conv3x3_bn, cifar=True, **kwargs):
    model = ResNet(BasicBlock, convBN, 18, num_classes, cifar, **kwargs)
    return model

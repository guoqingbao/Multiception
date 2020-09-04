'''
Author: Guoqing Bao
School of Computer Science, The University of Sydney
04/09/2020

Reference:
Guoqing Bao, Manuel B. Graeber, Xiuying Wang, "Depthwise Multiception Convolution for Reducing Network Parameters without Sacrificing Accuracy", 
16th International Conference on Control, Automation, Robotics and Vision (ICARCV 2020), In Press.

'''

import tensorflow as tf
import torch
import glob
import os
import numpy as np
from torchvision import datasets
from torchvision import transforms

def fetch_bylabel(label, dataset='cifar'):
    if dataset == 'cifar':
        if label == 10:
                normalizer = transforms.Normalize(mean= [0.4914, 0.4824, 0.4467],
                                                std= [0.2471, 0.2435, 0.2616])
                data_cls = datasets.CIFAR10
        else:
                normalizer = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                            std=[0.2675, 0.2565, 0.2761])
                data_cls = datasets.CIFAR100
    elif dataset == 'stl':
            normalizer = transforms.Normalize(mean= [0.4914, 0.4824, 0.4467],
                                            std= [0.2471, 0.2435, 0.2616])
            data_cls = datasets.STL10
    else:
            normalizer = None
            data_cls = None
    return normalizer, data_cls


def load_dataset(label, batch_size, data_path, dataset='cifar', download=False):
    normalizer, data_cls = fetch_bylabel(label, dataset)

    if dataset == 'cifar':
        train_loader = torch.utils.data.DataLoader(
            data_cls(data_path, train=True, download=download,
                     transform=transforms.Compose([
                         transforms.RandomCrop(32, padding=4),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         normalizer
                     ])),
            batch_size=batch_size, shuffle=True, num_workers=2)

        test_loader = torch.utils.data.DataLoader(
            data_cls(data_path, train=False, download=download,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         normalizer
                     ])),
            batch_size=batch_size, shuffle=False, num_workers=2)
    elif dataset == 'stl':
        train_loader = torch.utils.data.DataLoader(
            data_cls(data_path, split='train', download=download,
                     transform=transforms.Compose([
                         transforms.RandomCrop(96, padding=4),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         normalizer
                     ])),
            batch_size=batch_size, shuffle=True, num_workers=2)

        test_loader = torch.utils.data.DataLoader(
            data_cls(data_path, split='test', download=download,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         normalizer
                     ])),
            batch_size=batch_size, shuffle=False, num_workers=2)
    elif dataset =="imagenet32":
        from imagenet32dataset import ImageNet32Dataset
        dataset = ImageNet32Dataset(data_path + '/imagenet-tfr', True)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=128)
        dataset1 = ImageNet32Dataset(data_path + '/imagenet-tfr', False)
        test_loader = torch.utils.data.DataLoader(dataset1, batch_size=128)

    return train_loader, test_loader
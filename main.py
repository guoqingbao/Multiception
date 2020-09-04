'''
Author: Guoqing Bao
School of Computer Science, The University of Sydney
04/09/2020

Reference:
Guoqing Bao, Manuel B. Graeber, Xiuying Wang, "Depthwise Multiception Convolution for Reducing Network Parameters without Sacrificing Accuracy", 
16th International Conference on Control, Automation, Robotics and Vision (ICARCV 2020), In Press.

'''

import os
from os.path import expanduser
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import argparse

#please make sure pytorch 1.2 + was installed
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tqdm import tqdm

import utils

#dataset and models
from dataset import load_dataset
from shakenet import ShakeResNet
from senet import se_resnet56
from resnet import resnet20, resnet50
from mobilenet import MobileNetV2

#import three different convolution
from convolution import Multiception, depthwise_separable_conv, conv3x3_bn, MixConv
import numpy as np

#cosine learning rate
import math
def CosinePolicy(rate, t_cur, restart_period):
    return rate/2 * (1. + math.cos(math.pi *
                                (t_cur / restart_period)))

#choose different models using configured parameters
def main(args):
    #make sure cifar-10, cifar-100, STL-10 and Imagenet 32x32 downloaded in this path, 
    # otherwise change the download to True in load_dataset function
    path = expanduser("~") + "/project/data" #the path of the data
    train_loader, test_loader = load_dataset(args.label, args.batch_size, path, args.dataset, download=False)
    cifarlike = (args.dataset=='cifar') 
    if args.model =="shake":
        model = ShakeResNet(args.convBN, args.depth, args.factor, args.label, cifar=cifarlike)
    elif args.model == "senet56":
        model = se_resnet56(num_classes = args.label, reduction=8, convBN = args.convBN, cifar=cifarlike)
    elif args.model == 'resnet20':
        model = resnet20(num_classes = args.label, convBN = args.convBN, cifar=cifarlike)
    elif args.model == 'resnet50':
        model = resnet50(num_classes = args.label, convBN = args.convBN, cifar=cifarlike)
    elif args.model == 'mobilenet':
        if cifarlike and args.convBN.__name__=='conv3x3_bn':
            print("Small image input or standard convolution is not supported in MobileNet!")
            return
        model = MobileNetV2(num_classes = args.label, convBN = args.convBN, large=False)
    else:
        print("Invalid model!")
        return

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benckmark = True
    print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))

    opt = optim.SGD(model.parameters(),
                    lr=args.lr,
                    momentum=0.9,
                    weight_decay=args.weight_decay,
                    nesterov=args.nesterov)
    loss_func = nn.CrossEntropyLoss().cuda()

    headers = ["Epoch", "LearningRate", "TrainLoss", "TestLoss", "TrainAcc.", "TestAcc."]
    logger = utils.Logger(args.log_path, args.log_file, headers)

    #let's train and test the model
    for e in range(args.epochs):
        lr = utils.cosine_lr(opt, args.lr, e, args.epochs)
        model.train()
        train_loss, train_acc, train_n = 0, 0, 0
        for x, t in train_loader:
            x, t = Variable(x.cuda()), Variable(t.cuda())
            y = model(x)
            loss = loss_func(y, t)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_acc += utils.accuracy(y, t).item()
            train_loss += loss.item() * t.size(0)
            train_n += t.size(0)

        model.eval()
        test_loss, test_acc, test_n = 0, 0, 0
        for x, t in tqdm(test_loader, total=len(test_loader), leave=False):
            with torch.no_grad():
                x, t = Variable(x.cuda()), Variable(t.cuda())
                y = model(x)
                loss = loss_func(y, t)
                test_loss += loss.item() * t.size(0)
                test_acc += utils.accuracy(y, t).item()
                test_n += t.size(0)
        logger.write(e+1, lr, train_loss / train_n, test_loss / test_n,
                     train_acc / train_n * 100, test_acc / test_n * 100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=int, default=10)
    parser.add_argument("--log_path", type=str, default="./log/")
    parser.add_argument("--model", type=str, default="shake")

    parser.add_argument("--depth", type=int, default=26)
    parser.add_argument("--factor", type=int, default=96)
    parser.add_argument("--cardinary", type=int, default=4)

    parser.add_argument("--lr", type=float, default=0.1)

    #for fair comparison, dropout is not used for all models 
    # parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--weight_decay", type=float, default=0.0002)
    parser.add_argument("--nesterov", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)

    # cifar dataset by default
    parser.add_argument("--dataset", type=str, default="cifar")

    args = parser.parse_args()

    # run three times for each convolution type
    nets = []
    lognums = [1,2,3, 1,2,3, 1,2,3]
    types = [depthwise_separable_conv, depthwise_separable_conv, depthwise_separable_conv, Multiception, Multiception, Multiception, conv3x3_bn, conv3x3_bn, conv3x3_bn ]
    # types = [MixConv, MixConv, MixConv] #enable this for testing MixConv

    # model type accepted from arguments
    for i in range(len(types)):
        nets.append(args.model)

    for i in range(len(types)):
        args.model = nets[i]
        args.convBN = types[i]
        typename = types[i].__name__
        if nets[i] =='resnet20' or nets[i] =='resnet50' or nets[i]=='senet56' or nets[i]=='mobilenet':
            print("*******Start Job for ", args.model, ' -  Conv Type', typename)
        else:
            if nets[i] == 'shake': #ShakeNet-26x96d
                args.depth = 26
                args.factor = 96
            print("*******Start Job for ", args.model, " depth", args.depth, " factor:", args.factor, ' -  Conv Type', typename)

        
        convtype = 'Baseline' if typename=='conv3x3_bn' else typename # networks with conv3x3_bn (standard convolution) are baseline models

        if nets[i] =='resnet20' or nets[i] =='resnet50' or nets[i] =='senet56':
            args.log_file = nets[i] + '_'+ str(args.label) + '-' + convtype + '_' +  str(lognums[i]) + '.log'
        else:
            args.log_file = nets[i] + str(args.label) + '-' + convtype + '_' + str(lognums[i]) + '.log'


        if args.dataset == 'stl':
            args.log_file = 'stl_' + args.log_file
        elif args.dataset == 'imagenet32':
            args.log_file = 'imagenet32_' + args.log_file

        print("Log file: ", args.log_file)
        main(args)
        print('\r\n************Job Down for Model '+ nets[i] +'  Conv Type:', convtype, '  ***************\r\n')


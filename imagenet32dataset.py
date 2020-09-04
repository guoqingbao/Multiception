'''
Author: Guoqing Bao
School of Computer Science, The University of Sydney
04/09/2020

Reference:
Guoqing Bao, Manuel B. Graeber, Xiuying Wang, "Depthwise Multiception Convolution for Reducing Network Parameters without Sacrificing Accuracy", 
16th International Conference on Control, Automation, Robotics and Vision (ICARCV 2020), In Press.

'''

'''
Code to read imagenet 32x32 dataset from files, 
please make sure images downloaded in the corresponding directory
'''
import tensorflow as tf
import torch
import os

# tf.enable_eager_execution()
from torch.utils.data.dataset import Dataset 
from torchvision import transforms
from os.path import expanduser
import numpy as np
import os
import glob


class ImageNet32Dataset(Dataset):
    def __init__(self, path, train=True):
        self.train = train
        self.files = self.get_tfr_files(path, 'train' if train else 'validation')
        self.dataset, self.itr  = self.input_fn(self.files)
        self.data_len = (1280000 if train else 50000)
        self.trans=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                ])
        
    def get_tfr_files(self, path, split):
        path = os.path.join(path, split)
        tfr_prefix = os.path.join(path, os.path.basename(path))
        tfr_file = tfr_prefix + '-r%02d-s-*-of-*.tfrecords' % (5)
        print(tfr_file)
        files = glob.glob(tfr_file)
        return files
    
    def parse_tfrecord_tf(self, record):
        features = tf.io.parse_single_example(record, features={
            'shape': tf.io.FixedLenFeature([3], tf.int64),
            'data': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([1], tf.int64)})

        data, label, shape = features['data'], features['label'], features['shape']
        label = tf.cast(tf.reshape(label, shape=[]), dtype=tf.int32)
        img = tf.io.decode_raw(data, tf.uint8)
        img = tf.reshape(img, [32, 32, 3])
        # img = tf.transpose(img, [2,0,1])
        return img, label 


    def input_fn(self, tfr_file):
        filenames = tf.data.Dataset.list_files(tfr_file)
        if self.train:
            filenames = filenames.shuffle(buffer_size=1024)

        dset = filenames.apply(
            tf.data.experimental.parallel_interleave(lambda filename: tf.data.TFRecordDataset(filename), cycle_length=16))

        # dset = files.apply(tf.data.experimental.parallel_interleave(
        #     tf.data.TFRecordDataset, cycle_length=1))
        if self.train:
            dset = dset.shuffle(buffer_size=128*4)
        dset = dset.repeat()
        dset = dset.map(lambda x: self.parse_tfrecord_tf(x), num_parallel_calls=16)
        dset = dset.prefetch(256)
        itr = tf.compat.v1.data.make_one_shot_iterator(dset)
        # itr = dset.make_one_shot_iterator()
        return dset, itr

    def __getitem__(self, index):
        x, y = self.itr.get_next()
        return self.trans((x.numpy()/255).astype(np.float32)), y.numpy().astype(np.long)

    def __len__(self):
        return self.data_len
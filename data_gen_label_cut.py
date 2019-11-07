#!/usr/bin/env python
# coding: utf-8
"""
 @Time    : 19-9-21 上午10:45
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : data_gen_label_cut.py
"""

import math

import numpy as np
from PIL import Image

Image.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, Sequence
from sklearn.model_selection import train_test_split
import pandas as pd
from Cutup import cutup
from random_eraser import get_random_eraser

class BaseSequence(Sequence):
    """
    基础的数据流生成器，每次迭代返回一个batch
    BaseSequence可直接用于fit_generator的generator参数
    fit_generator会将BaseSequence再次封装为一个多进程的数据流生成器
    而且能保证在多进程下的一个epoch中不会重复取相同的样本
    """

    def __init__(self, img_paths, labels, batch_size, img_size, preprocess_input, use):
        assert len(img_paths) == len(labels), "len(img_paths) must equal to len(lables)"
        assert img_size[0] == img_size[1], "img_size[0] must equal to img_size[1]"
        self.x_y = np.hstack((np.array(img_paths).reshape(len(img_paths), 1), np.array(labels)))
        self.batch_size = batch_size
        self.img_size = img_size
        self.use = use
        self.alpha = 1.0
        self.preprocess_input = preprocess_input
        self.eraser = get_random_eraser(s_h=0.1, pixel_level=True)

    def __len__(self):
        return math.ceil(len(self.x_y) / self.batch_size)

    @staticmethod
    def resizeimg(img, size):
        img = img.resize((size, size), Image.ANTIALIAS)
        return img

    def preprocess_img(self, img_path):
        img = Image.open(img_path)
        resize_scale = self.img_size[0] / max(img.size[:2])
        img = img.resize((int(img.size[0] * resize_scale), int(img.size[1] * resize_scale)))
        img = img.convert('RGB')
        img = img.resize((self.img_size[0], self.img_size[0]), Image.ANTIALIAS)
        img = np.array(img)
        if self.use:
            # img = self.eraser(img)
            datagen = ImageDataGenerator(
                width_shift_range = 0.05,
                height_shift_range = 0.05,
                horizontal_flip = True,
            )
            img = datagen.random_transform(img)
        img = img[:, :, ::-1]
        return img

    def __getitem__(self, idx):

        batch_x = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 0]
        batch_y = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 1:]
        batch_x = np.array([self.preprocess_img(img_path) for img_path in batch_x])
        batch_y = np.array(batch_y).astype(np.float32)
        if self.use:
            axis = np.random.randint(low = 0, high = 10)
            if axis > 3:
                batch_x, batch_y = cutup(batch_x, batch_y)
            else:
                batch_x, batch_y = self.mixup(batch_x, batch_y)
        batch_x = self.preprocess_input(batch_x)

        return batch_x, batch_y

    def on_epoch_end(self):

        np.random.shuffle(self.x_y)

    def mixup(self, batch_x, batch_y):
        size = batch_x.shape[0]
        l = np.random.beta(self.alpha, self.alpha, size)

        X_l = l.reshape(size, 1, 1, 1)
        y_l = l.reshape(size, 1)

        X1 = batch_x
        Y1 = batch_y
        X2 = batch_x[::-1]
        Y2 = batch_y[::-1]

        X = X1 * X_l + X2 * (1 - X_l)
        Y = Y1 * y_l + Y2 * (1 - y_l)

        return X, Y


def smooth_labels(y, smooth_factor = 0.1):
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception(
            'Invalid label smoothing factor: ' + str(smooth_factor))
    return y


def data_flow(batch_size, num_classes, input_size, preprocess_input):
    #     train_data_dir='../Train_label.csv'
    train_data_dir = '../Train_label_new_8.csv'
    train_df = pd.read_csv(train_data_dir)
    tags = []
    train_df['type'] = train_df['type'].astype(str)
    for i in range(train_df['type'].shape[0]):
        for tag in train_df['type'].iloc[i].split(','):
            tag = int(tag) - 1
            tags.append(tag)
    print('total label: %d ' % len(tags))

    for i in range(train_df['type'].shape[0]):
        train_df['FileName'].iloc[i] = train_df['FileName'].iloc[i].split('/')[-1]
    img_paths = list(train_df['FileName'])
    local_path = '../Train_New/'
    for i in range(len(img_paths)):
        img_paths[i] = local_path + img_paths[i]
    print('total img: %d ' % (len(img_paths)))

    labels = np_utils.to_categorical(tags, num_classes)
    # 标签平滑
    labels = smooth_labels(labels)
    train_img_paths, validation_img_paths, train_labels, validation_labels = \
        train_test_split(img_paths, labels, test_size = 0.1, shuffle = True)
    print('total samples: %d, training samples: %d, validation samples: %d' % (
        len(img_paths), len(train_img_paths), len(validation_img_paths)))

    train_sequence = BaseSequence(train_img_paths, train_labels, batch_size, [input_size, input_size],
                                  preprocess_input, use = True)
    validation_sequence = BaseSequence(validation_img_paths, validation_labels, batch_size, [input_size, input_size],
                                       preprocess_input, use = False)
    return train_sequence, validation_sequence


if __name__ == '__main__':
    train_data_dir = '../Train_label.csv'
    batch_size = 8

    train_sequence, validation_sequence = data_flow(train_data_dir, batch_size, num_classes = 40, input_size = 224,
                                                    preprocess_input = preprocess_input)

    for i in range(2):
        print(i)
        batch_data, bacth_label = train_sequence.__getitem__(i)
        print(batch_data, bacth_label)
        # batch_data_, bacth_label_ = validation_sequence.__getitem__(i)
        # print(batch_data_,bacth_label_)

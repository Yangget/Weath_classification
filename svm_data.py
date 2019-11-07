#!/usr/bin/env python
# coding: utf-8
"""
 @Time    : 19-10-12 下午3:15
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : svm_data.py
"""

import numpy as np
import pandas as pd
from PIL import Image

Image.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def preprocess_img_(img_path, img_size):
    img = Image.open(img_path)
    resize_scale = img_size[0] / max(img.size[:2])
    img = img.resize((int(img.size[0] * resize_scale), int(img.size[1] * resize_scale)))
    img = img.convert('RGB')
    img = img.resize((img_size[0], img_size[0]), Image.ANTIALIAS)
    img = np.array(img)
    img = img[:, :, ::-1]
    return img


def load_test_data_train(input_size, preprocess_input):
    train_data_url = '../Train_svm.csv'
    #     train_data_url='../Train_label_new.csv'
    train_df = pd.read_csv(train_data_url)
    train_dir = []
    train_img_paths = []
    tags = []
    train_df['type'] = train_df['type'].astype(str)
    for i in range(train_df['type'].shape[0]):
        for tag in train_df['type'].iloc[i].split(','):
            tag = int(tag) - 1
            tags.append(tag)
    print('total label: %d ' % len(tags))
    train_label = tags

    for i in range(len(train_df['type'])):
        train_img_paths.append(train_df['FileName'].iloc[i].split('/')[-1])
    train_local_path = '../Train_New/'
    train_data = []
    import os
    for i in range(len(train_img_paths)):
        train_data.append(preprocess_img_(os.path.join(train_local_path, train_img_paths[i]), (input_size, input_size)))
    print('total img: %d ' % (len(train_img_paths)))
    train_data = np.array(train_data)
    train_data = preprocess_input(train_data)
    print('Train data ok!')
    return train_data, train_label

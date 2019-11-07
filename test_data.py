#!/usr/bin/env python
# coding: utf-8
"""
 @Time    : 19-9-20 上午8:30
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : test_data.py
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


def load_test_data(input_size, preprocess_input):
    test_data_url = '../test_A.csv'
    test_df = pd.read_csv(test_data_url)
    test_dir = []
    img_paths = []
    for i in range(4260):
        img_paths.append(test_df['FileName'].iloc[i].split('/')[-1])
    local_path = '../Test/'
    test_data = []
    import os
    for i in range(len(img_paths)):
        test_data.append(preprocess_img_(os.path.join(local_path, img_paths[i]), (input_size, input_size)))
    print('total img: %d ' % (len(img_paths)))
    test_data = np.array(test_data)
    test_data = preprocess_input(test_data)
    print('Test data ok!')
    return img_paths, test_data

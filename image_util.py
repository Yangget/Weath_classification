#!/usr/bin/env python
# coding: utf-8
"""
 @Time    : 19-9-15 上午11:05
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : image_util.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def preprocess_input(x):
    x = x.astype(np.float32)
    x /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    x[..., 0] /= std[0]
    x[..., 1] /= std[1]
    x[..., 2] /= std[2]
    return x

#!/usr/bin/env python
# coding: utf-8
"""
 @Time    : 19-9-23 上午6:55
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : Cutup.py
"""

import numpy as np


def get_rand_bbox(width, height, l):
    r_x = np.random.randint(width)
    r_y = np.random.randint(height)
    r_l = np.sqrt(1 - l)
    r_w = np.int(width * r_l)
    r_h = np.int(height * r_l)
    bb_x_1 = np.clip(r_x - r_w // 2, 0, width)
    bb_y_1 = np.clip(r_y - r_h // 2, 0, height)
    bb_x_2 = np.clip(r_x + r_w // 2, 0, width)
    bb_y_2 = np.clip(r_y + r_h // 2, 0, height)
    return bb_x_1, bb_y_1, bb_x_2, bb_y_2


def cutup(batch_x, batch_y):
    #     print("cutup")
    alpha = 1.0
    _, h, w, c = batch_x.shape
    #     print(batch_x.shape)
    #     print(batch_x_2.shape)

    lam = np.random.beta(alpha, alpha)

    X1 = batch_x
    Y1 = batch_y
    X2 = batch_x[::-1]
    Y2 = batch_y[::-1]
    bx1, by1, bx2, by2 = get_rand_bbox(w, h, lam)
    X1[:, bx1:bx2, by1:by2, :] = X2[:, bx1:bx2, by1:by2, :]
    X = X1
    y = Y1 * lam + Y2 * (1 - lam)

    return X, y

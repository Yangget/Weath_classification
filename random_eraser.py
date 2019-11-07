#!/usr/bin/env python
# coding: utf-8
"""
 @Time    : 19-10-13 下午2:00
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : random_eraser.py
"""

import numpy as np


def get_random_eraser(p = 0.8, s_l = 0.02, s_h = 0.4, r_1 = 0.3, r_2 = 1 / 0.3, v_l = 0, v_h = 255,
                      pixel_level = False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand( )

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser

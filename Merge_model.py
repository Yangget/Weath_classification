#!/usr/bin/env python
# coding: utf-8
"""
 @Time    : 19-10-26 上午7:50
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : Merge_model.py
"""

import numpy as np
import pandas as pd
import scipy.stats as ss

label = []
###############################################
model1 = pd.read_csv(r'./submit/submit_1.csv')
predict1 = np.asarray(model1['type'])
label.append(predict1)
###############################################
model2 = pd.read_csv(r'./submit/submit_2.csv')
predict2 = np.asarray(model2['type'])
label.append(predict2)
################################################
model3 = pd.read_csv(r'./submit/submit_3.csv')
predict3 = np.asarray(model3['type'])
label.append(predict3)
################################################
model4 = pd.read_csv(r'./submit/submit_4.csv')
predict4 = np.asarray(model4['type'])
label.append(predict4)
##################################################
model5 = pd.read_csv(r'./submit/submit_5.csv')
predict5 = np.asarray(model5['type'])
label.append(predict5)

pred = label
pred = np.array(pred)
pred = np.transpose(pred, (1, 0))
pred = ss.mode(pred, axis = 1)[0]
predictions = np.squeeze(pred)
print(predictions)

sample_submission_df = pd.read_csv("../test_A.csv")
sample_submission_df['type'] = predictions
sample_submission_df.to_csv('./result/test_fin.csv', index = None)

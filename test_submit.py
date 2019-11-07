#!/usr/bin/env python
# coding: utf-8
"""
 @Time    : 19-11-6 上午9:20
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : test_submit.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import numpy as np

from keras import backend
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

backend.set_image_data_format('channels_last')

from PIL import Image

Image.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1
import efficientnet.keras as efn

preprocess_input = efn.preprocess_input
## 2
# seresnext50, preprocess_input = Classifiers.get('seresnext50')
## 3
# xception, preprocess_input = Classifiers.get('xception')
## 4
# densenet201, preprocess_input = Classifiers.get('densenet201')
## 5
# inceptionresnetv2, preprocess_input = Classifiers.get('inceptionresnetv2')

input_size = 380
num_classes = 9
tta_steps = 5


def model_fn(objective, optimizer, metrics):
    base_model = efn.EfficientNetB4(include_top = False,
                                    # base_model = seresnext50(include_top=False,
                                    # base_model = xception(include_top=False,
                                    # base_model = densenet201(include_top=False,
                                    # base_model = inceptionresnetv2(include_top=False,
                                    input_shape = (input_size, input_size, 3),
                                    classes = num_classes,
                                    weights = 'imagenet', )
    x = base_model.output
    x = GlobalAveragePooling2D( )(x)
    predictions = Dense(9, activation = 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    model.compile(loss = objective, optimizer = optimizer, metrics = metrics)
    model.summary( )
    return model


optimizer = Adam(lr = 0.01)
objective = 'categorical_crossentropy'
metrics = ['accuracy']
model1 = model_fn(objective, optimizer, metrics)

model1.load_weights('./model_snapshots/X0001/weights-improvement-44-0.8930.h5')
# model1.load_weights('./model_snapshots/X0002/weights-improvement-49-0.8977.h5')
# model1.load_weights('./model_snapshots/X0003/weights-improvement-49-0.8874.h5')
# model1.load_weights('./model_snapshots/X0004/weights-improvement-48-0.9002.h5')
# model1.load_weights('./model_snapshots/X0005/weights-improvement-42-0.8846.h5')


# 1,3,4,5
intermediate_model1 = Model(inputs = model1.input, outputs = [model1.get_layer('global_average_pooling2d_1').output])
# # 2
# intermediate_model1 = Model(inputs=model1.input, outputs=[model1.get_layer('global_average_pooling2d_17').output])

from test_data import load_test_data

img_paths, test_data = load_test_data(input_size, preprocess_input)

from svm_data import load_test_data_train

train_data, train_label = load_test_data_train(input_size, preprocess_input)

print('###################  intermediate_model train_going  ##############################')
from keras.preprocessing.image import ImageDataGenerator

test_datagen1 = ImageDataGenerator(
    horizontal_flip = True,
)
predictions_1 = []
for i in range(tta_steps):
    preds = intermediate_model1.predict_generator(test_datagen1.flow(train_data, batch_size = bs, shuffle = False),
                                                  steps = len(train_data) / bs, verbose = 1)
    predictions_1.append(preds)
intermediate_output = np.mean(predictions_1, axis = 0)

print('###################  SVM train_going  ##############################')
from sklearn.svm import SVC

svm = SVC(kernel = 'rbf', gamma = 'auto')
svm.fit(intermediate_output, train_label)

print('###################  intermediate_model test_going  ##############################')
predictions_1 = []
for i in range(tta_steps):
    preds = intermediate_model1.predict_generator(test_datagen1.flow(test_data, batch_size = bs, shuffle = False),
                                                  steps = len(test_data) / bs, verbose = 1)
    predictions_1.append(preds)
final_pred = np.mean(predictions_1, axis = 0)

print('###################  SVM test_going  ##############################')
pre_fin = svm.predict(final_pred)

print('###################  wirte submit  ##############################')
infos = []
for index, pred_ in enumerate(pre_fin):
    pred_label = pred_ + 1
    infos.append('%s,%s\n' % (img_paths[index], pred_label))

path = './submit/submit_1.csv'
# path = './submit/submit_2.csv'
# path = './submit/submit_3.csv'
# path = './submit_test_4.csv'
# path = './submit/submit_5.csv'

with open(path, 'w') as f:
    f.write('FileName,type\n')
    f.writelines(infos)
print('end')

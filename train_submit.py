#!/usr/bin/env python
# coding: utf-8
"""
 @Time    : 19-11-6 上午8:50
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : train_submit.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import warnings

warnings.filterwarnings("ignore")
from PIL import Image

Image.LOAD_TRUNCATED_IMAGES = True
import multiprocessing

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras_radam import RAdam
from lookahead import Lookahead
from data_gen_label_cut import data_flow

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


# effb4 batch_size = 38
epochs = 50
lrate = 1e-3
batch_size = 16
num_classes = 9
input_size = 380
train_sequence, validation_sequence = data_flow(batch_size, num_classes, input_size, preprocess_input)


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
    predictions = Dense(num_classes, activation = 'softmax')(x)
    model1 = Model(inputs = base_model.input, outputs = predictions)
    #     model2 = multi_gpu_model(model1, gpus=3)
    #     model2 = model1
    model1.compile(loss = objective, optimizer = optimizer, metrics = metrics)
    lookahead = Lookahead(k = 5, alpha = 0.5)  # Initialize Lookahead
    lookahead.inject(model1)  # add into model
    model1.summary( )
    return model1


optimizer = RAdam(learning_rate = lrate)
objective = 'categorical_crossentropy'
metrics = ['accuracy']
model1 = model_fn(objective, optimizer, metrics)
print("model ok!")

filepath = './model_snapshots/X0001/weights-improvement-{epoch:02d}-{val_acc:.4f}.h5'
# filepath = './model_snapshots/X0002/weights-improvement-{epoch:02d}-{val_acc:.4f}.h5'
# filepath = './model_snapshots/X0003/weights-improvement-{epoch:02d}-{val_acc:.4f}.h5'
# filepath = './model_snapshots/X0004/weights-improvement-{epoch:02d}-{val_acc:.4f}.h5'
# filepath = './model_snapshots/X0005/weights-improvement-{epoch:02d}-{val_acc:.4f}.h5'
cbk1 = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True, save_weights_only = True,
                       mode = 'auto',
                       period = 1)

# log_local = './logs/X0001-effb4(380)-cutmix&mixup(1.0)-wp'
# tensorBoard = TensorBoard(log_dir=log_local)

from warmup_cosine_decay_scheduler import WarmUpCosineDecayScheduler

sample_count = len(train_sequence) * batch_size
epochs = epochs
warmup_epoch = 8
batch_size = batch_size
learning_rate_base = lrate
total_steps = int(epochs * sample_count / batch_size)
warmup_steps = int(warmup_epoch * sample_count / batch_size)

warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base = learning_rate_base,
                                        total_steps = total_steps,
                                        warmup_learning_rate = 0,
                                        warmup_steps = warmup_steps,
                                        hold_base_rate_steps = 0,
                                        )

model1.fit_generator(
    train_sequence,
    steps_per_epoch = len(train_sequence),
    epochs = epochs,
    verbose = 1,
    callbacks = [cbk1, warm_up_lr],
    validation_data = validation_sequence,
    max_queue_size = 10,
    workers = int(multiprocessing.cpu_count( ) * 0.9),
    use_multiprocessing = True,
    shuffle = True
)

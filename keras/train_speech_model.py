import os
import time
import pandas as pd
from utils import data
import keras
from speech_model import build_speech_model
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import numpy as np
from sgdr import *
from sklearn.utils.class_weight import compute_class_weight
from keras import regularizers
import tensorflow as tf

TRAIN_INC = False

trainset = pd.read_csv('/home/dmp/tuan/speech_keras/data/train.csv')
valset = pd.read_csv('/home/dmp/tuan/speech_keras/data/validation.csv')
background_set = pd.read_csv('/home/dmp/tuan/speech_keras/data/background.csv')
batch_size = 128
num_epochs = 70
nb_classes = 12

learning_rate = 0.1
momentum = 0.9
weight_decay = 0.005

optimiz = keras.optimizers.SGD(lr = learning_rate, momentum = momentum, nesterov = False)

schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=0.1,
                                     steps_per_epoch=np.ceil(num_epochs/batch_size),
                                     lr_decay=weight_decay,
                                     cycle_length=10,
                                     mult_factor=2)
                                     
train = data.AudioDataset(trainset, background_set)
validation = data.AudioDataset(valset)

def cal_class_weights(y_integers):
  class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
  d_class_weights = dict(enumerate(class_weights))
  return d_class_weights

def input_generator(train):
  inputs = []
  labels = []
  j = 0
  while True:
    idx = data.get_sampler(train)
    for i in idx:
      if(j == 0):
        inputs.clear()
        labels.clear()
      if(j < batch_size):
        inputs.append(train[i]['sound'])
        labels.append(to_categorical(train[i]['label'], nb_classes))
        j += 1
      else:
        j = 0
        binputs = np.array(inputs)
        blabels = np.array(labels)
        yield (binputs, blabels)
      
model = build_speech_model(inputs=(16000, 1), num_classes=nb_classes)
if(TRAIN_INC == True):
  model.load_weights('w_speech_net.hdf5')

for layer in model.layers:
    if hasattr(layer, 'kernel_regularizer'):
        layer.kernel_regularizer= regularizers.l2(weight_decay)
model.summary()

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
callbacks_list = [checkpoint, schedule]


steps = len(train)/batch_size
training_set = input_generator(train)
validation_set = input_generator(validation)

def my_loss(y_pred, y_true):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_true, labels=y_pred))
    
model.compile(loss = my_loss, optimizer = optimiz, metrics=['accuracy'])
model.fit_generator(training_set, verbose = 1, epochs = num_epochs, validation_data = validation_set, validation_steps=10, 
                              steps_per_epoch = steps, callbacks = callbacks_list)

model.save_weights('w_speech_net.hdf5')
model.save('speech_net.h5')
print("save model done ...")

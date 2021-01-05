from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dropout,
    Lambda,
    GlobalAveragePooling1D,
    ZeroPadding1D,
    Dense
)
from keras.layers.convolutional import (
    Conv1D,
    MaxPooling1D,
    AveragePooling1D
)
from keras.layers.normalization import BatchNormalization
import tensorflow as tf


def make_block(filters, kernel_size=5, padding=2, mp_kernel_size=4):
  def f(inputs):
      x = Conv1D(filters, kernel_size=kernel_size, padding='same')(inputs)
      x = BatchNormalization()(x)
      x = Activation("relu")(x)
      x = Conv1D(filters, kernel_size=kernel_size, padding='same')(x)
      x = BatchNormalization()(x)
      x = Activation("relu")(x)
      if mp_kernel_size:
          x = MaxPooling1D(pool_size=mp_kernel_size)(x)
      return x
  return f
        

def build_speech_model(inputs=(16000, 1), num_classes=12):
  input_net = Input(shape=inputs)
  net = make_block(8)(input_net)
  net = make_block(16)(net)
  net = make_block(32)(net)
  net = make_block(64)(net)
  net = make_block(128)(net)
  net = make_block(256, mp_kernel_size=None)(net)
  net = GlobalAveragePooling1D()(net)
  net = Dense(128)(net)
  net = Activation("relu")(net)
  net = Dropout(0.5)(net)
  net = Dense(64)(net)
  net = Activation("relu")(net)
  net = Dropout(0.5)(net)
  net = Dense(num_classes)(net)
  
  model = Model(inputs=input_net, outputs=net)
  
  return model
  
#model = build_speech_model(inputs=(16000, 1), num_classes=12)
#model.summary()
  
  
  
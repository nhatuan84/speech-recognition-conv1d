import os
import time
import pandas as pd
from utils import data
import keras
from speech_model import build_speech_model
from keras.utils import to_categorical
import numpy as np
from keras.models import load_model



nb_classes = 12

testset = pd.read_csv('/home/dmp/tuan/speech_keras/data/test.csv')
test = data.AudioPredictionDataset(testset)
        
      
model = build_speech_model(inputs=(16000, 1), num_classes=nb_classes)
model.load_weights('w_speech_net.hdf5')

outputs = []
for i in range(len(test)):
  output = model.predict(test[i]['sound'])
  predictied = np.argmax(output, axis=1)
  outputs.append({'filename': test[i]['filename'], 'predicted': predictied})
  
print(outputs)

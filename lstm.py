# -*- coding: utf-8 -*-
# !/usr/bin/env python

import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import json
import matplotlib.pyplot as plt

csv_path = 'USD_JPY_Week1.csv'

df = pd.read_csv(csv_path)

# extract RateBid value
data = df['RateBid'].astype(np.float32)

# normalize
data_norm = (data - data.mean()) / data.std()
#data_norm = (data - data.min()) / (data.max() - data.min())

"""
1. z-score normalization
x_new = (x - x_mean) / x_std
外れ値にもロバスト
standardized_sample_df = (datasample_df - sample_df.mean()) / sample_df.std()
print(standardized_sample_df)
"""

"""
2. min-max normalization
0から1になるように変換
x_new = (x - x_min) / (x_max - x_min)
normalized_sample_df = (sample_df - sample_df.min()) / (sample_df.max() - sample_df.min())
print(normalized_sample_df)
"""

x, y = [], []
N = len(data_norm)
M = 1*14*60
L = 1*14*60

for i in range(M,N-L):
  _x = data_norm[i-M:i]
  _y = data_norm[i+L]

  x.append(_x)
  y.append(_y)

# 70%訓練用データ 30%テストデータ
N_train = int(N*0.7)
x_train, x_test = x[:N_train], x[N_train:]
y_train, y_test = y[:N_train], y[N_train:]

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

model = keras.Sequential()
model.add(keras.layers.CuDNNLSTM(30,recurrent_initializer='glorot_normal'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1,activation='linear'))
model.compile(loss='mean_squared_error',
              optimizer='Adam',
              metrics=['mse'])

epoch=3
batch_size=1000

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1],1))
y_train = y_train.reshape((y_train.shape[0],1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],1))
y_test = y_test.reshape((y_test.shape[0],1))

yen_history = model.fit(x_train,y_train,
          epochs=epoch,
          batch_size=batch_size,
          validation_data=(x_test,y_test))

model.summary()

fig, ax1 = plt.subplots(1,1)
ax1.plot(yen_history.epoch, yen_history.history['loss'])
ax1.set_title('Training Error')

if model.loss == 'mae':
    ax1.set_ylabel('Mean Absolute Error (MAE)',fontsize=12)
else:
    ax1.set_ylabel('Model Loss',fontsize=12)
ax1.set_xlabel('# Epochs',fontsize=12)
plt.show()


predict = model.predict(x_test)
plt.plot(y_test * data.std() + data.mean(), label='real price')
plt.plot(predict * data.std() + data.mean(), label='predicted price' )
plt.legend()
plt.show()
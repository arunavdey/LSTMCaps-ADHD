# import tensorflow as tf
# import os
# from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Sequential


ann = Sequential()
ann.add(Input())
ann.add(Dense(units=512, activation='relu'))
ann.add(Dense(units=1024, activation='relu'))
ann.add(Dense(units=512, activation='relu'))
ann.add(Dense(units=1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='mse')
ann.fit(x=None, y=None, batch_size=10, epochs=50)

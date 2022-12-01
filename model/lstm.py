import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import callbacks
from keras.layers import (LSTM, BatchNormalization,
                          Dense, Dropout, Embedding, Input, LeakyReLU)
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Model
from keras.optimizers import Adam
from utils import load_data_csv


def get_model(shape, xlen):

    # (x_train, y_train), (x_test, y_test) = load_data_csv(site)

    x = Input(shape=shape[1:])
    embedding1 = Embedding(
        xlen, 4, input_length=shape[1:])(x)
    lstm1 = LSTM(64, return_sequences=True)(embedding1)
    lstm2 = LSTM(64, return_sequences=True)(lstm1)
    lstm3 = LSTM(64)(lstm2)
    lrelu1 = LeakyReLU(alpha = 0.1)(lstm3)
    batch_norm = BatchNormalization()(lrelu1)
    dense1 = Dense(64)(batch_norm)
    lrelu2 = LeakyReLU(alpha = 0.1)(dense1)
    dropout1 = Dropout(0.2)(lrelu2)
    dense2 = Dense(64)(dropout1)
    lrelu3 = LeakyReLU(alpha = 0.1)(dense2)
    batch_norm = BatchNormalization()(lrelu3)
    dense3 = Dense(4, activation='softmax')(dense2)

    model = Model(inputs=[x], outputs=[dense3])

    return model

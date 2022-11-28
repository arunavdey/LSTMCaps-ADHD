import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import callbacks
from keras.layers import (LSTM, BatchNormalization,
                          Dense, Dropout, Embedding, Input)
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Model
from keras.optimizers import Adam
from utils import load_data_csv

sites = ["KKI", "Peking_1", "Peking_2", "Peking_3"]

(x_train, y_train), (x_test, y_test) = load_data_csv(sites[0])

def get_model():

    print(x_train.shape)

    x = Input(shape=x_train.shape[1:])
    embedding1 = Embedding(
        len(x_train), 4, input_length=x_train.shape[1:])(x)
    lstm1 = LSTM(256, return_sequences=True)(embedding1)
    lstm2 = LSTM(256, return_sequences=True)(lstm1)
    lstm3 = LSTM(256, return_sequences=True)(lstm2)
    lstm4 = LSTM(128, activation='relu')(lstm3)
    batch_norm = BatchNormalization()(lstm4)
    dense1 = Dense(64, activation='relu')(batch_norm)
    dropout1 = Dropout(0.2)(dense1)
    dense2 = Dense(64, activation='relu')(dropout1)
    batch_norm = BatchNormalization()(lstm4)
    dense3 = Dense(4, activation='softmax')(dense2)

    model = Model(inputs=[x], outputs=[dense3])

    return model

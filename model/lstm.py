import keras
import numpy as np
import pandas as pd
from sklearn.metrics._classification import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import BatchNormalization, LSTM, Dense, Dropout, Embedding
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def load_data():
    adhd = pd.read_csv("../feature-extraction/features/KKI_adhd_func.csv")
    control = pd.read_csv("../feature-extraction/features/KKI_control_func.csv")

    data = pd.concat([adhd, control])

    x = data.iloc[1:, 0:-1]
    y = data.iloc[1:, -1]

    ss = StandardScaler()
    x = ss.fit_transform(x)

    x = x.astype('float32') / 255.0

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0)

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()

model = Sequential()
model.add(Embedding(len(x_train), 64, input_length = x_train.shape[1]))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(64, input_dim = 12, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss=SparseCategoricalCrossentropy(), optimizer=Adam(learning_rate = 1e-6), metrics=['accuracy'] )

print(model.summary())

model.fit(x_train, y_train, epochs=10, batch_size = 32, validation_data=(x_test,y_test))

y_pred = model.predict(x_test)

test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

model.save("./saved-models/lstm2.h5")

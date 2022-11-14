import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import callbacks
from keras.layers import (LSTM, BatchNormalization,
                          Dense, Dropout, Embedding, Input)
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Model
from keras.optimizers import Adam
# from utils import plot_log


def load_data():
    adhd = pd.read_csv(
        "../feature-extraction/features/KKI_adhd_func.csv")
    control = pd.read_csv(
        "../feature-extraction/features/KKI_control_func.csv")

    data = pd.concat([adhd, control])

    x = data.iloc[1:, 0:-1]
    y = data.iloc[1:, -1]
    # x = data.iloc[1:2048, 0:-1]
    # y = data.iloc[1:2048, -1]

    ss = StandardScaler()
    x = ss.fit_transform(x)

    x = x.astype('float32') / 255.0
    y = y.astype('float32')

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0)

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()


def get_model():

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
    dense3 = Dense(4, activation='softmax')(dense2)

    model = Model(inputs=[x], outputs=[dense3])

    log = callbacks.CSVLogger('./logs/lstm_logs.csv')
    checkpoint = callbacks.ModelCheckpoint('./weights/lstm_weights-{epoch:02d}.h5', monitor='val_accuracy',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    model.compile(loss=SparseCategoricalCrossentropy(),
                  optimizer=Adam(learning_rate=0.1), metrics=['accuracy'])

    model.summary()

    model.fit(x_train, y_train, epochs=5, batch_size=2048,
              validation_data=(x_test, y_test), callbacks=[log, checkpoint])

    model.save_weights("./saved-models/lstm_trained.h5")
    print(f"Trained model saved to \'%s./saved-models/lstm-trained.h5\'")

    # plot_log('./logs/lstm_train_log.csv', show=True)

    y_pred = model.predict(x_test)

    return y_pred


if __name__ == "__main__":

    model = get_model()

    y_pred = model.predict(x_test)

    test_loss, test_acc = model.evaluate(x_test, y_test)

    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_acc}")

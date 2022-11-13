import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import (LSTM, BatchNormalization,
                          Dense, Dropout, Embedding)
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Sequential
from keras.optimizers import Adam


def load_data():
    adhd = pd.read_csv(
        "../feature-extraction/features/KKI_adhd_func.csv")
    control = pd.read_csv(
        "../feature-extraction/features/KKI_control_func.csv")

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


def get_model():

    # TODO
    # rewrite as a functional model

    model = Sequential()

    model.add(Embedding(len(x_train), 8, input_length=x_train.shape[1]))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64, input_dim=12, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss=SparseCategoricalCrossentropy(),
                  optimizer=Adam(learning_rate=0.1), metrics=['accuracy'])

    model.summary()

    model.fit(x_train, y_train, epochs=10, batch_size=2048,
              validation_data=(x_test, y_test))

    model.save_weights("./saved-models/lstm_trained.h5")
    print(f"Trained model saved to \'%s/saved-models/lstm-trained.h5\'")

    return model


if __name__ == "__main__":

    model = get_model()

    y_pred = model.predict(x_test)

    test_loss, test_acc = model.evaluate(x_test, y_test)

    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_acc}")

import keras
import os
import numpy as np
import pandas as pd
from capsulenet import get_model as getCapsNet
from capsulenet import euclidean_distance_loss, margin_loss
from capsulenet import get_args
from lstm import get_model as getLSTMNet
from utils import load_data_csv, load_data_mri
from keras import callbacks, layers, models, optimizers, losses
from scikeras.wrappers import KerasClassifier

dir_home = os.path.join("/mnt", "hdd")
dir_adhd200 = os.path.join(dir_home, "Assets", "ADHD200")
sites = ["KKI", "Peking_1", "Peking_2", "Peking_3"]

def trainCapsNet(site):
    model = getCapsNet()
    (x_train, y_train), (x_test, y_test) = load_data_mri(site)
    args = get_args()

    log = callbacks.CSVLogger(f'./logs/{site}_capsnet_logs.csv')

    checkpoint = callbacks.ModelCheckpoint(
            f'./weights/{site}_capsnet_weights.h5', monitor='val_out_caps_accuracy', 
            save_best_only=True, save_weights_only=True, verbose=1)

    lr_decay = callbacks.LearningRateScheduler(
        schedule=lambda epoch: args.learning_rate * (args.lr_decay ** epoch))

    model.compile(optimizer=optimizers.Adam(learning_rate=args.learning_rate),
                  loss=[margin_loss, euclidean_distance_loss],
                  loss_weights=[1., args.lam_recon],
                  metrics={'out_caps': 'accuracy'})

    model.fit(x=[x_train, y_train], y=[y_train, x_train], batch_size=2, epochs=10,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, checkpoint, lr_decay])


    model.save_weights(f'./saved-models/{site}_capsnet_trained.h5')

    y_pred, x_recon = model.predict([x_test, y_test], batch_size=2)

    return y_pred

def trainLSTMNet(site):
    model = getLSTMNet()
    (x_train, y_train), (x_test, y_test) = load_data_csv(site)

    log = callbacks.CSVLogger(f'./logs/{site}_lstm_logs.csv')

    checkpoint = callbacks.ModelCheckpoint(f'./weights/{site}_lstm_weights.h5', monitor='val_accuracy',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    model.compile(loss=losses.SparseCategoricalCrossentropy(),
                  optimizer=optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])


    model.fit(x_train, y_train, epochs=10, batch_size=1024,
              validation_data=(x_test, y_test), callbacks=[log, checkpoint])

    model.save_weights(f"./saved-models/{site}_lstm_trained.h5")

    y_pred = model.predict(x_test)

    return y_pred


if __name__ == "__main__":

    adhd = {0: "No ADHD", 1: "ADHD 1", 2: "ADHD 2", 3 : "ADHD 3"}

    site = sites[0]
    subs = []

    pheno = pd.read_csv(os.path.join(dir_adhd200, f"{site}_athena", f"{site}_preproc", f"{site}_phenotypic.csv"))
    adhdRows = pd.read_csv(f"../feature-extraction/features/{site}_adhd_rows_func.csv")
    controlRows = pd.read_csv(f"../feature-extraction/features/{site}_control_rows_func.csv")

    # subID, number of rows
    rows = pd.concat([adhdRows, controlRows]).to_numpy() # shape: all subs, 2

    capsnet = trainCapsNet(site)
    lstmnet = trainLSTMNet(site)

    pd.DataFrame(capsnet).to_csv("result/pred_caps.csv", index = False)
    pd.DataFrame(lstmnet).to_csv("result/pred_lstm.csv", index = False) 

    new_pred_lstm = np.empty((4))
    temp_lstmnet = lstmnet
    flag = 0

    for sub, numRows in rows[::-1]:
        if not flag:
            shape = temp_lstmnet.shape
            if shape[0] < numRows:
                numRows = shape[0]
                flag = 1

            preds = np.mean(temp_lstmnet[-numRows:], axis = 0)
            temp_lstmnet = temp_lstmnet[:-numRows] 
            new_pred_lstm = np.vstack([preds, new_pred_lstm])
        else:
            break

    new_pred_lstm = new_pred_lstm[:-2]

    final_pred = np.empty((4))

    for idx, pred in enumerate(new_pred_lstm):
        p = [(pred[0] + pred_caps[idx][0])/2, (pred[1] + pred_caps[idx][1])/2, (pred[2] + pred_caps[idx][2])/2, (pred[3] + pred_caps[idx][3])]

        final_pred = np.vstack([final_pred, p])

    final_pred = final_pred[1:]

    for pred in final_pred:
        print(adhd[np.argmax(pred)])

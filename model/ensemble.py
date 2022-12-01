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
# sites = ["KKI", "Peking_1", "Peking_2", "Peking_3"]
sites = ["KKI"]
adhd = {0: "No ADHD", 1: "ADHD 1", 2: "ADHD 2", 3 : "ADHD 3"}

def trainCapsNet(site):

    lr = 1e-4

    (x_train, y_train), (x_test, y_test) = load_data_mri(site)
    args = get_args()

    model, eval_model = getCapsNet(x_train.shape)

    log = callbacks.CSVLogger(f'./logs/{site}_capsnet_logs.csv')

    checkpoint = callbacks.ModelCheckpoint(
            f'./weights/{site}_capsnet_weights.h5', monitor='val_out_caps_accuracy', 
            save_best_only=True, save_weights_only=True, verbose=1)

    lr_decay = callbacks.LearningRateScheduler(
        schedule=(lambda epoch: lr * (0.8 ** epoch)))

    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'out_caps': 'accuracy'})

    print(model.summary())

    model.fit(x=[x_train, y_train], y=[y_train, x_train], batch_size=2, epochs=8,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, checkpoint, lr_decay])


    model.save_weights(f'./saved-models/{site}_capsnet_trained.h5')

    y_pred, x_recon = model.predict([x_test, y_test], batch_size = 2)

    return y_pred

def trainLSTMNet(site):

    lr = 1e-2

    (x_train, y_train), (x_test, y_test) = load_data_csv(site)

    model = getLSTMNet(x_train.shape, len(x_train))

    log = callbacks.CSVLogger(f'./logs/{site}_lstm_logs.csv')

    checkpoint = callbacks.ModelCheckpoint(
            f'./weights/{site}_lstm_weights.h5', monitor='val_accuracy',
            save_best_only=True, save_weights_only=True, verbose=1)

    model.compile(loss=losses.SparseCategoricalCrossentropy(),
    # model.compile(loss=losses.CategoricalCrossentropy(axis = 0),
                  optimizer=optimizers.Adam(learning_rate=lr), metrics=['accuracy'])

    print(model.summary())


    model.fit(x_train, y_train, epochs=8, batch_size=256,
              validation_data=(x_test, y_test), callbacks=[log, checkpoint])

    model.save_weights(f"./saved-models/{site}_lstm_trained.h5")

    y_pred = model.predict(x_test)

    return y_pred


if __name__ == "__main__":


    for site in sites:
        pheno = pd.read_csv(os.path.join(dir_adhd200, f"{site}_athena", f"{site}_preproc", f"{site}_phenotypic.csv"))
        rows = pd.read_csv(f"../feature-extraction/features/{site}_rows_func.csv").to_numpy()

        capsnet = trainCapsNet(site)
        pd.DataFrame(capsnet).to_csv(f"result/{site}_pred_caps3.csv", index = False)

        lstmnet = trainLSTMNet(site)
        pd.DataFrame(lstmnet).to_csv(f"result/{site}_pred_lstm3.csv", index = False) 

        # capsnet = pd.read_csv(f"result/{site}_pred_caps2.csv").to_numpy()
        # lstmnet = pd.read_csv(f"result/{site}_pred_lstm2.csv").to_numpy()

        # capsnet = pd.read_csv(f"result/KKI_pred_caps2.csv").to_numpy()
        # lstmnet = pd.read_csv(f"result/KKI_pred_lstm2.csv").to_numpy()

        new_pred_lstm = np.empty((4))
        new_pred_capsnet = np.empty((4))
        temp_lstmnet = lstmnet
        temp_capsnet = capsnet

        print(capsnet.shape)
        print(lstmnet.shape)

        for idx in range(0, len(capsnet), 16):
            preds = np.mean(temp_capsnet[:16], axis = 0)
            temp_capsnet = temp_capsnet[16:]
            new_pred_capsnet = np.vstack([new_pred_capsnet, preds])

        new_pred_capsnet = new_pred_capsnet[2:]

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

        new_pred_lstm = new_pred_lstm[:-1]

        # print(new_pred_capsnet.shape)
        # print(new_pred_lstm.shape)

        final_pred = np.empty((4))

        for idx in range(len(new_pred_lstm)):
            p = [(new_pred_lstm[idx][0] + new_pred_capsnet[idx][0])/2, (new_pred_lstm[idx][1] + new_pred_capsnet[idx][1])/2, (new_pred_lstm[idx][2] + new_pred_capsnet[idx][2])/2, (new_pred_lstm[idx][3] + new_pred_capsnet[idx][3])/2] 
            final_pred = np.vstack([final_pred, p])

        final_pred = final_pred[1:]

        final_pred = [np.argmax(row) for row in final_pred]

        y_true = pheno["DX"]
        
        total = 83
        i = len(final_pred) - 1

        corpred = 0
        mispred = 0

        while i > 0:
            if final_pred[i] == 0 and y_true[total - i] == 0:
                corpred += 1
            elif final_pred[i] == 1 and y_true[total - i] != 1:
                corpred += 1
            elif final_pred[i] == 2 and y_true[total - i] != 2:
                corpred += 1
            elif final_pred[i] == 3 and y_true[total - i] != 3:
                corpred += 1
            else:
                mispred += 1
            i -= 1
        
        acc = (corpred) / (corpred + mispred)

        print(f"number of test subjects: {len(final_pred)}")
        print(f"correct predictions: {corpred}")
        print(f"mispredictions: {mispred}")
        print(f"combined acc: {acc}")

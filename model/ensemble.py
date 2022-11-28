import keras
import numpy as np
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

def train_capsnet(site):
    model = getCapsNet()
    (x_train, y_train), (x_test, y_test) = load_data_mri(site)
    args = get_args()

    log = callbacks.CSVLogger(f'./logs/{site}_capsnet_logs.csv')
    checkpoint = callbacks.ModelCheckpoint(f'./weights/{site}_capsnet_weights-{epoch:02d}.h5', monitor='val_out_caps_accuracy',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(
        schedule=lambda epoch: args.learning_rate * (args.lr_decay ** epoch))
    model.compile(optimizer=optimizers.Adam(learning_rate=args.learning_rate),
                  loss=[margin_loss, euclidean_distance_loss],
                  loss_weights=[1., args.lam_recon],
                  metrics={'out_caps': 'accuracy'})

    model.fit(x=[x_train, y_train], y=[y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, checkpoint, lr_decay])


    model.save_weights(f'./saved-models/{site}_capsnet_trained.h5')

    y_pred, x_recon = model.predict([x_test, y_test], batch_size=2)

    return y_pred

def train_lstmnet(site):
    model = getLSTMNet()
    (x_train, y_train), (x_test, y_test) = load_data_csv(site)

    log = callbacks.CSVLogger(f'./logs/{site}_lstm_logs.csv')
    checkpoint = callbacks.ModelCheckpoint(f'./weights/{site}_lstm_weights-{epoch:02d}.h5', monitor='val_accuracy',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    model.compile(loss=losses.SparseCategoricalCrossentropy(),
                  optimizer=optimizers.Adam(learning_rate=0.1), metrics=['accuracy'])

    # model.summary()

    model.fit(x_train, y_train, epochs=1, batch_size=2048,
              validation_data=(x_test, y_test), callbacks=[log, checkpoint])

    model.save_weights(f"./saved-models/{site}_lstm_trained.h5")

    y_pred = model.predict(x_test)

    return y_pred


if __name__ == "__main__":


    for site in sites:
        subs = []

        pheno = pd.read_csv(os.path.join(dir_adhd200, f"{site}_athena", f"{site}_preproc", f"{site}_phenotypic.csv"))
    
        for ind in tqdm.tqdm(pheno.index):
            subID = pheno["ScanDir ID"][ind]
            subs.append(subID)

        pred_caps = train_capsnet(site)
        pred_lstm = train_lstmnet(site)

    # read the csv for rows for particular site
    # rows have final index of each subject, can use to figure out 
    #   which rows of the pred_lstm belong to which subject (have to have matching dimensions
    #   of both pred_caps and pred_lstm).


    # print(pred_caps.shape) # 16, 4; num of test scans, output shape
    # print(pred_lstm.shape) # 410, 4; num of test rows, output shape



    # models = [getCapsNet(), getLSTMNet()]

    # # lstmnet = KerasClassifier(getLSTMNet, epochs=1, verbose=True)

    # inputs = [layers.Input(shape=()), layers.Input(shape=x_train.shape[1:])]
    # inputCaps = 
    # inputLSTM = 
    
    # ensemble_output = layers.Average()(models)
    # ensemble_model = keras.Model(inputs=models, outputs=ensemble_output)

    # ensemble_model.summary()

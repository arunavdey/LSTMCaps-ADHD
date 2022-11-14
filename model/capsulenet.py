import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from capsulelayers import CapsuleLayer, Length, Mask, PrimaryCap
from keras import backend as K
from keras import callbacks, layers, models, optimizers
from keras.utils import to_categorical
# from utils import plot_log

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings, batch_size):

    # generating inverse graphics
    x = layers.Input(shape=input_shape, batch_size=batch_size)
    conv1 = layers.Conv3D(filters=128, kernel_size=11, strides=1,
                          padding='valid', activation='relu', name='conv1')(x)
    pool1 = layers.AveragePooling3D(
        pool_size=2, padding='valid', name='pool1')(conv1)
    primarycaps = PrimaryCap(inputs=pool1, dim_capsule=32, n_channels=4,
                             kernel_size=11, strides=2, padding='valid')
    capslayer = CapsuleLayer(num_capsule=n_class, dim_capsule=32,
                             routings=routings, name='CapsLayer')(primarycaps)
    out_caps = Length(name='out_caps')(capslayer)

    # decoder network
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([capslayer, y])
    masked = Mask()(capslayer)
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=32 * n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation
    train_model = models.Model(inputs=[x, y], outputs=[
                               out_caps, decoder(masked_by_y)])
    eval_model = models.Model(inputs=x, outputs=[out_caps, decoder(masked)])

    return train_model, eval_model


def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.5 * \
        (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return tf.reduce_mean(tf.reduce_sum(L, 1))


def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def train(model, data, args):
    (x_train, y_train), (x_test, y_test) = data

    log = callbacks.CSVLogger('./logs/capsnet_logs.csv')
    checkpoint = callbacks.ModelCheckpoint('./weights/capsnet_weights-{epoch:02d}.h5', monitor='val_out_caps_accuracy',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(
        schedule=lambda epoch: args.learning_rate * (args.lr_decay ** epoch))

    model.compile(optimizer=optimizers.Adam(learning_rate=args.learning_rate),
                  loss=[margin_loss, euclidean_distance_loss],
                  loss_weights=[1., args.lam_recon],
                  metrics={'out_caps': 'accuracy'})

    model.fit(x=[x_train, y_train], y=[y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, checkpoint, lr_decay])

    model.save_weights('./saved-models/capsnet_trained.h5')
    print("Trained model saved to \'%s./saved-models/capsnet_trained.h5\'")

    # plot_log('./logs/capsnet_train_log.csv', show=True)

    y_pred, x_recon = model.predict([x_test, y_test], batch_size=2)

    return y_pred


def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-' * 30 + 'Begin: test' + '-' * 30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1)
          == np.argmax(y_test, 1)) / y_test.shape[0])


def load_data():
    dir_home = os.path.join("/mnt", "d")
    kki_athena = os.path.join(dir_home, "Assets", "ADHD200", "KKI_athena")

    kki_pheno_path = os.path.join(
        kki_athena, "KKI_preproc", "KKI_phenotypic.csv")
    kki_pheno = pd.read_csv(kki_pheno_path)

    kki_preproc = os.path.join(kki_athena, "KKI_preproc")

    kki_subs = kki_pheno["ScanDir ID"].to_numpy()

    x = list()

    for sub in tqdm.tqdm(kki_subs, desc='Loading kki'):
        scan_path = os.path.join(
            kki_preproc, f"{sub}", f"wmean_mrda{sub}_session_1_rest_1.nii.gz")

        scan = nib.load(scan_path).get_fdata()
        x.append(scan)

    x = np.array(x)

    y = kki_pheno["DX"].to_numpy()

    print(x.shape)
    print(y.shape)

    x_train, x_test = x[:64], x[64:80]
    y_train, y_test = y[:64], y[64:80]

    # x_train = x_train.reshape(-1, 197, 233, 189, 1).astype('float32') / 255.
    x_train = x_train.reshape(-1, 49, 58, 47, 1).astype('float32') / 255.

    # x_test = x_test.reshape(-1, 197, 233, 189, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 49, 58, 47, 1).astype('float32') / 255.

    y_train = to_categorical(y_train.astype('float32'))

    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)


def get_model():
    parser = argparse.ArgumentParser(description="Capsule Network")
    parser.add_argument('-e', '--epochs', default=10, type=int)
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--lr_decay', default=0.95, type=float)
    parser.add_argument('--lam_recon', default=0.9, type=float)
    parser.add_argument('-r', '--routings', default=4, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true')
    parser.add_argument('-w', '--weights', default=None)
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    (x_train, y_train), (x_test, y_test) = load_data()

    model, eval_model = CapsNet(input_shape=x_train.shape[1:],
                                n_class=4,
                                routings=args.routings,
                                batch_size=args.batch_size)
    model.summary()

    if args.weights is not None:
        model.load_weights(args.weights)
    if not args.testing:
        return train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:
        test(model=eval_model, data=(x_test, y_test), args=args)


if __name__ == "__main__":
    get_model()

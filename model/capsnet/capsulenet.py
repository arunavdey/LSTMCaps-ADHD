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
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings, batch_size):

    # input_shape   : shape of scan (4D for 3D scan)
    # n_class       : number of target classes
    # routings      : number of times to run the routing algorithm
    # batch_size    : the batch size

    x = layers.Input(shape=input_shape, batch_size=batch_size)

    # pool1 = layers.AveragePooling3D(pool_size=2, padding='valid')(x)

    conv1 = layers.Conv3D(filters=128, kernel_size=11, strides=1,
                          padding='valid', activation='relu', name='conv1')(x)

    pool1 = layers.AveragePooling3D(
        pool_size=2, padding='valid', name='pool1')(conv1)

    # pool2 = layers.AveragePooling3D(
    #     pool_size=2, padding='valid')(pool1)

    primarycaps = PrimaryCap(inputs=pool1, dim_capsule=32, n_channels=4,
                             kernel_size=11, strides=2, padding='valid')

    capslayer = CapsuleLayer(num_capsule=n_class, dim_capsule=32,
                             routings=routings, name='CapsLayer')(primarycaps)

    out_caps = Length(name='out_caps')(capslayer)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([capslayer, y])
    masked = Mask()(capslayer)

    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=32 * n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
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

    # early_stop = callbacks.EarlyStopping(
    #     monitor='out_caps_loss', mode='auto')
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_out_caps_accuracy',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(
        schedule=lambda epoch: args.learning_rate * (args.lr_decay ** epoch))

    model.compile(optimizer=optimizers.Adam(learning_rate=args.learning_rate),
                  loss=[margin_loss, euclidean_distance_loss],
                  loss_weights=[1., args.lam_recon],
                  metrics={'out_caps': 'accuracy'})

    model.fit(x=[x_train, y_train], y=[y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, checkpoint, lr_decay])

    # def train_generator(x, y, batch_size, shift_fraction=0.):
    #     train_datagen = ImageDataGenerator(
    #         width_shift_range=shift_fraction, height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
    #     generator = train_datagen.flow(
    #         x[:, :, :, :, 0], y, batch_size=batch_size)
    #     while 1:
    #         x_batch, y_batch = generator.next()
    #         yield (x_batch, y_batch), (y_batch, x_batch)

    # # Training with data augmentation. If shift_fraction=0., no augmentation.
    # model.fit(train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
    #           steps_per_epoch=int(y_train.shape[0] / args.batch_size),
    #           epochs=args.epochs,
    #           validation_data=((x_test, y_test), (y_test, x_test)
    #                            ), batch_size=args.batch_size,
    #           callbacks=[log, checkpoint, lr_decay])

    model.save_weights(args.save_dir + '/capsnet_trained.h5')
    print('Trained model saved to \'%s/capsnet_trained.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-' * 30 + 'Begin: test' + '-' * 30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1)
          == np.argmax(y_test, 1)) / y_test.shape[0])


def load_data():
    dir_home = os.path.join("/mnt", "d")
    kki_athena = os.path.join(dir_home, "Assets", "ADHD200", "KKI_athena")
    peking1_athena = os.path.join(
        dir_home, "Assets", "ADHD200", "Peking_1_athena")
    peking2_athena = os.path.join(
        dir_home, "Assets", "ADHD200", "Peking_2_athena")
    peking3_athena = os.path.join(
        dir_home, "Assets", "ADHD200", "Peking_3_athena")

    # kki_pheno_path = os.path.join(
    #     kki_athena, "KKI_preproc", "KKI_phenotypic.csv")
    # kki_pheno = pd.read_csv(kki_pheno_path)

    peking1_pheno_path = os.path.join(
        peking1_athena, "Peking_1_preproc", "Peking_1_phenotypic.csv")
    peking1_pheno = pd.read_csv(peking1_pheno_path)
    peking2_pheno_path = os.path.join(
        peking2_athena, "Peking_2_preproc", "Peking_2_phenotypic.csv")
    peking2_pheno = pd.read_csv(peking2_pheno_path)
    peking3_pheno_path = os.path.join(
        peking3_athena, "Peking_3_preproc", "Peking_3_phenotypic.csv")
    peking3_pheno = pd.read_csv(peking3_pheno_path)

    # kki_fmri_path = os.path.join(kki_athena, "KKI_preproc")
    peking1_fmri_path = os.path.join(peking1_athena, "Peking_1_preproc")
    peking2_fmri_path = os.path.join(peking2_athena, "Peking_2_preproc")
    peking3_fmri_path = os.path.join(peking3_athena, "Peking_3_preproc")

    # kki_subs = kki_pheno["ScanDir ID"].to_numpy()
    peking1_subs = peking1_pheno["ScanDir ID"].to_numpy()
    peking2_subs = peking2_pheno["ScanDir ID"].to_numpy()
    peking3_subs = peking3_pheno["ScanDir ID"].to_numpy()

    x = list()

    # for sub in tqdm.tqdm(kki_subs, desc='Loading kki'):
    #     scan_path = os.path.join(
    #         kki_fmri_path, f"{sub}", f"wmean_mrda{sub}_session_1_rest_1.nii.gz")
    #     scan = nib.load(scan_path).get_fdata()
    #     x.append(scan)

    for sub in tqdm.tqdm(peking1_subs, desc='Loading peking 1'):
        scan_path = os.path.join(
            peking1_fmri_path, f"{sub}", f"wmean_mrda{sub}_session_1_rest_1.nii.gz")
        scan = nib.load(scan_path).get_fdata()
        x.append(scan)

    for sub in tqdm.tqdm(peking2_subs, desc='Loading peking 2'):
        scan_path = os.path.join(
            peking2_fmri_path, f"{sub}", f"wmean_mrda{sub}_session_1_rest_1.nii.gz")
        scan = nib.load(scan_path).get_fdata()
        x.append(scan)

    for sub in tqdm.tqdm(peking3_subs, desc='Loading peking 3'):
        scan_path = os.path.join(
            peking3_fmri_path, f"{sub}", f"wmean_mrda{sub}_session_1_rest_1.nii.gz")
        scan = nib.load(scan_path).get_fdata()
        x.append(scan)

    x = np.array(x)
    y = peking1_pheno["DX"].to_numpy()
    # y = np.append(y, peking1_pheno["DX"].to_numpy())
    y = np.append(y, peking2_pheno["DX"].to_numpy())
    y = np.append(y, peking3_pheno["DX"].to_numpy())

    # for i in range(y.size):
    #     if y[i] > 0:
    #         y[i] = 1

    print(x.shape)
    print(y.shape)

    x_train, x_test = x[:174], x[174:]
    y_train, y_test = y[:174], y[174:]

    # x_train, x_test = x[:250], x[250:270]
    # y_train, y_test = y[:250], y[250:270]

    x_train = x_train.reshape(-1, 49, 58, 47, 1).astype('float32') / 255.
    # x_train = x_train.reshape(-1, 49, 58, 47, 1).astype('float32')
    x_test = x_test.reshape(-1, 49, 58, 47, 1).astype('float32') / 255.
    # x_test = x_test.reshape(-1, 49, 58, 47, 1).astype('float32')
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Capsule Network")
    parser.add_argument('-e', '--epochs', default=20, type=int)
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('--learning_rate', default=1e-6, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.65, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=1e-5, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=4, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
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
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:
        test(model=eval_model, data=(x_test, y_test), args=args)

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from capsulelayers import CapsuleLayer, Length, Mask, PrimaryCap
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings, batch_size):

    x = layers.Input(shape=input_shape, batch_size=batch_size)
    # x = layers.Input(shape=input_shape)
    conv1 = layers.Conv3D(filters=128, kernel_size=9, strides=1,
                          padding='valid', activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(
        inputs=conv1, dim_capsule=32, n_channels=8, kernel_size=9, strides=2, padding='valid')
    capslayer = CapsuleLayer(num_capsule=n_class, dim_capsule=32,
                             routings=routings, name='capslayer')(primarycaps)
    out_caps = Length(name='out_caps')(capslayer)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([capslayer, y])
    masked = Mask()(capslayer)

    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=32 * n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model(inputs=[x, y], outputs=[
                               out_caps, decoder(masked_by_y)])
    eval_model = models.Model(inputs=x, outputs=[out_caps, decoder(masked)])

    return train_model, eval_model


def margin_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + 0.5 * \
        (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

    return tf.reduce_mean(tf.reduce_sum(L, 1))


def euclidean_distance_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def train(model, data, args):
    (x_train, y_train), (x_test, y_test) = data

    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(
        schedule=lambda epoch: args.learning_rate * (args.lr_decay ** epoch))

    model.compile(optimizer=optimizers.Adam(learning_rate=args.learning_rate),
                  loss=[margin_loss, euclidean_distance_loss],
                  loss_weights=[1., args.lam_recon],
                  metrics={'out_caps': 'accuracy'})

    model.fit(x=[x_train, y_train], y=[y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, checkpoint, lr_decay])

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

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

    kki_pheno_path = os.path.join(
        kki_athena, "KKI_preproc", "KKI_phenotypic.csv")
    kki_pheno = pd.read_csv(kki_pheno_path)

    peking1_pheno_path = os.path.join(
        peking1_athena, "Peking_1_preproc", "Peking_1_phenotypic.csv")
    peking1_pheno = pd.read_csv(peking1_pheno_path)

    kki_fmri_path = os.path.join(kki_athena, "KKI_preproc")
    peking1_fmri_path = os.path.join(peking1_athena, "Peking_1_preproc")

    kki_subs = kki_pheno["ScanDir ID"].to_numpy()
    peking1_subs = peking1_pheno["ScanDir ID"].to_numpy()

    x = list()

    for sub in tqdm.tqdm(kki_subs, desc='Loading kki'):
        scan_path = os.path.join(
            kki_fmri_path, f"{sub}", f"wmean_mrda{sub}_session_1_rest_1.nii.gz")
        scan = nib.load(scan_path).get_fdata()
        x.append(scan)

    for sub in tqdm.tqdm(peking1_subs, desc='Loading peking 1'):
        scan_path = os.path.join(
            peking1_fmri_path, f"{sub}", f"wmean_mrda{sub}_session_1_rest_1.nii.gz")
        scan = nib.load(scan_path).get_fdata()
        x.append(scan)

    x = np.array(x)
    y = kki_pheno["DX"].to_numpy()
    y = np.append(y, peking1_pheno["DX"].to_numpy())

    for i in range(y.size):
        if y[i] > 0:
            y[i] = 1

    x_train, x_test = x[:155], x[155:160]
    y_train, y_test = y[:155], y[155:160]

    x_train = x_train.reshape(-1, 49, 58, 47, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 49, 58, 47, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import argparse
    import os

    from tensorflow.keras import callbacks

    # from tensorflow.keras.preprocessing.image import ImageDataGenerator
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('-e', '--epochs', default=25, type=int)
    parser.add_argument('-b', '--batch_size', default=5, type=int)
    parser.add_argument('--learning_rate', default=0.0000001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.95, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=1, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    # parser.add_argument('--digit', default=5, type=int,
    #                     help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = load_data()

    # define model
    model, eval_model = CapsNet(input_shape=x_train.shape[1:],
                                n_class=2,
                                routings=args.routings,
                                batch_size=args.batch_size)
    model.summary()
    eval_model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test(model=eval_model, data=(x_test, y_test), args=args)

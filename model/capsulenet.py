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
from utils import load_data_mri

# K.set_image_data_format('channels_last')
sites = ["KKI", "Peking_1", "Peking_2", "Peking_3"]


def CapsNet(input_shape, n_class, routings, batch_size):

    # generating inverse graphics
    x = layers.Input(shape=input_shape, batch_size=batch_size)
    conv1 = layers.Conv3D(filters=64, kernel_size=15, strides=1,
                          padding='valid', kernel_initializer='glorot_normal', name='conv1')(x)
    # conv2 = layers.Conv3D(filters=16, kernel_size=5, strides=1,
    #                       padding='valid', name='conv2')(conv1)
    # conv3 = layers.Conv3D(filters=16, kernel_size=5, strides=1,
    #                       padding='valid', name='conv3')(conv2)
    lrelu1 = layers.LeakyReLU(alpha = 0.1)(conv1)
    dropout1 = layers.Dropout(0.3)(lrelu1)
    primarycaps = PrimaryCap(inputs=dropout1, dim_capsule=16, n_channels=n_class,
                             kernel_size=13, strides=2, padding='valid')
    capslayer = CapsuleLayer(num_capsule=n_class, dim_capsule=16,
                             routings=routings, name='CapsLayer')(primarycaps)

    out_caps = Length(name='out_caps')(capslayer)

    # decoder network
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([capslayer, y])
    masked = Mask()(capslayer)
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(64, input_dim=16 * n_class))
    decoder.add(layers.LeakyReLU(alpha = 0.1))
    decoder.add(layers.Dense(64))
    decoder.add(layers.LeakyReLU(alpha = 0.1))
    decoder.add(layers.Dense(64))
    decoder.add(layers.LeakyReLU(alpha = 0.1))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.Dense(np.prod(input_shape), activation='softmax'))
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


# def train(model, data, args):
#     (x_train, y_train), (x_test, y_test) = data

#     log = callbacks.CSVLogger('./logs/capsnet_logs.csv')
#     checkpoint = callbacks.ModelCheckpoint('./weights/capsnet_weights-{epoch:02d}.h5', monitor='val_out_caps_accuracy',
#                                            save_best_only=True, save_weights_only=True, verbose=1)
#     lr_decay = callbacks.LearningRateScheduler(
#         schedule=lambda epoch: args.learning_rate * (args.lr_decay ** epoch))

#     model.compile(optimizer=optimizers.Adam(learning_rate=args.learning_rate),
#                   loss=[margin_loss, 'mse'],
#                   loss_weights=[1., args.lam_recon],
#                   metrics={'out_caps': 'accuracy'})

#     model.fit(x=[x_train, y_train], y=[y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
#               validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, checkpoint, lr_decay])

#     model.save_weights('./saved-models/capsnet_trained.h5')
#     print("Trained model saved to \'%s./saved-models/capsnet_trained.h5\'")

#     # plot_log('./logs/capsnet_train_log.csv', show=True)

#     y_pred, x_recon = model.predict([x_test, y_test], batch_size=2)

#     return y_pred


# def test(model, data, args):
#     x_test, y_test = data
#     y_pred, x_recon = model.predict(x_test, batch_size=100)
#     print('-' * 30 + 'Begin: test' + '-' * 30)
#     print('Test acc:', np.sum(np.argmax(y_pred, 1)
#           == np.argmax(y_test, 1)) / y_test.shape[0])


def get_args():
    parser = argparse.ArgumentParser(description="Capsule Network")
    parser.add_argument('-e', '--epochs', default=1, type=int)
    parser.add_argument('-b', '--batch_size', default=4, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--lr_decay', default=.9, type=float)
    parser.add_argument('--lam_recon', default=.9, type=float)
    parser.add_argument('-r', '--routings', default=4, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true')
    parser.add_argument('-w', '--weights', default=None)
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    args = parser.parse_args()

    return args


def get_model(shape):
    args = get_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model, eval_model = CapsNet(input_shape=shape[1:],
                                n_class=4,
                                routings=args.routings,
                                batch_size=args.batch_size)

    return model, eval_model

    # if args.weights is not None:
    #     model.load_weights(args.weights)
    # if not args.testing:
    #     return train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    # else:
    #     test(model=eval_model, data=(x_test, y_test), args=args)

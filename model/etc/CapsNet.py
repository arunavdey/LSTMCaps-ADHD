import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tensorflow.keras.layers import Conv2D, Conv3D, Dense, Flatten, Input, Lambda, Layer, Reshape
from tensorflow.keras import backend as K
from tensorflow.keras import Model, initializers

def squash(inputs):
    squared_norm = K.sum(K.square(inputs), axis = -1, keepdims = True)
    return ((squared_norm/(1+squared_norm))/(K.sqrt(squared_norm+K.epsilon())))*inputs

class CapsuleLayer(Layer):
    def __init__(self, **kwargs):
        super(MyCustomLayer, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get('glorot_uniform')

    def build(self, input_shape):
        self.W = self.add_weight(shape = [2, 17 * 21 * 16 * 16, 8, 16],
                initializer = self.kernel_initializer,
                name = "weights")
        self.built = True

    def call(self, inputs):
        inputs = K.expand_dims(inputs, dim = 1)
        inputs = K.tile(inputs, [1, 2, 1, 1])
        inputs = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems = inputs)
        b = tf.zeros(shape = [K.shape(inputs)[0], 2, 17 * 21 * 16 * 16])

        for i in range(3-1):
            c = tf.nn.softmax(b, axis = 1)
            s = K.batch_dot(c, inputs, [2, 2])
            v = squash(s)
            b = b + K.batch_dot


dir_home = os.path.join("/mnt", "d")
dir_athena = os.path.join(dir_home, "Assets", "ADHD200", "kki_athena")
dir_niak = os.path.join(dir_home, "Assets", "ADHD200", "kki_niak")

subject = 1018959


fmri_input = Input(shape = (49, 58, 47, 1), name = 'fmri_input')
conv1 = Conv3D(256, (9, 9, 9), activation = 'relu', padding = 'valid', name = "conv1")(fmri_input)
conv2 = Conv3D(256, (9, 9, 9), strides = 2, padding = 'valid', name = "conv2")(conv1)
reshaped = Reshape((17 * 21 * 16 * 16, 16), name = "reshaped")(conv2)
squashed = Lambda(squash, name = "squashed")(reshaped)

model = Model(inputs = fmri_input, outputs = squashed)
print(model.summary())

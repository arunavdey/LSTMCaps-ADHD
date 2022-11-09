import keras
from keras.models import Model
from keras.layers import Conv2D, Dense, Input, Reshape, Lambda, Layer, Flatten
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
import tensorflow as tf
from keras import initializers
from keras.utils import to_categorical
from keras.layers.core import Activation

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# (x_train, y_orig_train), (x_test, y_orig_test) = mnist.load_data()
# x_train = x_train.astype('float32') / 255.
# x_train = x_train.reshape(-1,28,28,1)
# y_train = np.array(to_categorical(y_orig_train.astype('float32')))

# x_test = x_test.astype('float32') / 255.
# x_test = x_test.reshape(-1,28,28,1)
# y_test = np.array(to_categorical(y_orig_test.astype('float32')))

# x_output = x_train.reshape(-1,784)
# X_valid_output = x_test.reshape(-1,784)

# n_samples = 5

# plt.figure(figsize=(n_samples * 2, 3))
# for index in range(n_samples):
#     plt.subplot(1, n_samples, index + 1)
#     sample_image = x_test[index].reshape(28, 28)
#     plt.imshow(sample_image, cmap="binary")
#     plt.title("Label:" + str(y_orig_test[index]))
#     plt.axis("off")

# plt.show()

# input_shape = Input(shape=(28,28,1))  # size of input image is 28*28
 
# # a convolution layer output shape = 20*20*256
# conv1 = Conv2D(256, (9,9), activation = 'relu', padding = 'valid')(input_shape)

# # convolution layer with stride 2 and 256 filters of size 9*9
# conv2 = Conv2D(256, (9,9), strides = 2, padding = 'valid')(conv1)
 
# # reshape into 1152 capsules of 8 dimensional vectors
# reshaped = Reshape((6*6*32,8))(conv2)
 
# def squash(inputs):
#     # take norm of input vectors
#     squared_norm = K.sum(K.square(inputs), axis = -1, keepdims = True)
 
#     # use the formula for non-linear function to return squashed output
#     return ((squared_norm/(1+squared_norm))/(K.sqrt(squared_norm+K.epsilon())))*inputs

# # squash the reshaped output to make length of vector b/w 0 and 1
# squashed_output = Lambda(squash)(reshaped)

# class DigitCapsuleLayer(Layer):
#     # creating a layer class in keras
#     def __init__(self, **kwargs):
#         super(DigitCapsuleLayer, self).__init__(**kwargs)
#         self.kernel_initializer = initializers.get('glorot_uniform')
    
#     def build(self, input_shape): 
#         # initialize weight matrix for each capsule in lower layer
#         self.W = self.add_weight(shape = [10, 6*6*32, 16, 8], initializer = self.kernel_initializer, name = 'weights')
#         self.built = True
    
#     def call(self, inputs):
#         inputs = K.expand_dims(inputs, 1)
#         inputs = K.tile(inputs, [1, 10, 1, 1])

#         # matrix multiplication b/w previous layer output and weight matrix
#         inputs = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs)
#         b = tf.zeros(shape = [K.shape(inputs)[0], 10, 6*6*32])
        
# # routing algorithm with updating coupling coefficient c, using scalar product b/w input capsule and output capsule
#         for i in range(3-1):
#             c = tf.nn.softmax(b, dim = 1)
#             s = K.batch_dot(c, inputs, [2, 2])
#             v = squash(s)
#             b = b + K.batch_dot(v, inputs, [2, 3])
            
#         return v 

#     def compute_output_shape(self, input_shape):
#         return tuple([None, 10, 16])
    
    
    
# def output_layer(inputs):
#     return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())
 
# digit_caps = DigitCapsuleLayer()(squashed_output)
# outputs = Lambda(output_layer)(digit_caps)

# def mask(outputs):
 
#     if type(outputs) != list:  # mask at test time
#         norm_outputs = K.sqrt(K.sum(K.square(outputs), -1) + K.epsilon())
#         y  = K.one_hot(indices=K.argmax(norm_outputs, 1), num_classes = 10)
#         y = Reshape((10,1))(y)
#         return Flatten()(y*outputs)
 
#     else:    # mask at train time
#         y = Reshape((10,1))(outputs[1])
#         masked_output = y*outputs[0]
#         return Flatten()(masked_output)
    
# inputs = Input(shape = (10,))
# masked = Lambda(mask)([digit_caps, inputs])
# masked_for_test = Lambda(mask)(digit_caps)
 
# decoded_inputs = Input(shape = (16*10,))
# dense1 = Dense(512, activation = 'relu')(decoded_inputs)
# dense2 = Dense(1024, activation = 'relu')(dense1)
# decoded_outputs = Dense(784, activation = 'sigmoid')(dense2)
# decoded_outputs = Reshape((28,28,1))(decoded_outputs)

# def loss_fn(y_true, y_pred):
 
#     L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
 
#     return K.mean(K.sum(L, 1))

# decoder = Model(decoded_inputs, decoded_outputs)
# model = Model([input_shape,inputs],[outputs,decoder(masked)])
# test_model = Model(input_shape,[outputs,decoder(masked_for_test)])
 
# m = 128
# epochs = 10
# model.compile(optimizer=keras.optimizers.Adam(lr=0.001),loss=[loss_fn,'mse'],loss_weights = [1. ,0.0005],metrics=['accuracy'])
# model.fit([x_train, y_train],[y_train,x_train], batch_size = m, epochs = epochs, validation_data = ([x_test, y_test],[y_test,x_test]))
 
# label_predicted, image_predicted = model.predict([x_test, y_test])

# n_samples = 5

# plt.figure(figsize=(n_samples * 2, 3))
# for index in range(n_samples):
#     plt.subplot(1, n_samples, index + 1)
#     sample_image = x_test[index].reshape(28, 28)
#     plt.imshow(sample_image, cmap="binary")
#     plt.title("Label:" + str(y_orig_test[index]))
#     plt.axis("off")

# plt.show()

# plt.figure(figsize=(n_samples * 2, 3))
# for index in range(n_samples):
#     plt.subplot(1, n_samples, index + 1)
#     sample_image = image_predicted[index].reshape(28, 28)
#     plt.imshow(sample_image, cmap="binary")
#     plt.title("Predicted:" + str(np.argmax(label_predicted[index])))
#     plt.axis("off")

# plt.show()

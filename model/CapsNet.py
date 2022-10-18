import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt 


class CapsuleLayer:
    convolution = tf.keras.layers.Conv2D(256, [9, 9], strides = [1, 1], name = 'ConvolutionLayer', activation = 'relu')
    primaryCapsule = tf.keras.layers.Conv2D(32 * 8, [9, 9], strides = [2, 2], name = 'PrimaryCapsuleLayer')




def CapsNet(shape):
    # layer 1:  conv layer, filter 9x9, stride 1
    # layer 2:  conv layer, filter 9x9, stride 2
    # layer 3:  dynamic routing layer
    # layer 4:  flatten


if __name__ == "__main__":
    print("Capsule Network")

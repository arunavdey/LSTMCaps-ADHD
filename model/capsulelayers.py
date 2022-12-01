import keras.backend as K
import tensorflow as tf
from keras import initializers, layers

class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(tf.reduce_sum(K.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Mask(layers.Layer):
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            assert len(inputs) == 2
            inputs, mask = inputs
        else:
            x = K.sqrt(tf.reduce_sum(K.square(inputs), -1))
            mask = tf.one_hot(indices=tf.argmax(x, 1), depth=x.shape[1])

        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config


def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(
        K.square(vectors), axis=axis, keepdims=True)

    scale = (s_squared_norm / (1 + s_squared_norm)) / \
        K.sqrt(s_squared_norm + K.epsilon())

    return scale * vectors


class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(
            input_shape) >= 3, "Input tensor should have shape = [None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 1), -1)
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1, 1])
        inputs_hat = tf.squeeze(
            K.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled))

        # Routing algo
        b = tf.zeros(
            shape=[inputs.shape[0], self.num_capsule, 1, self.input_num_capsule])

        outputs = None

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            # c = tf.nn.sigmoid(b)
            outputs = squash(tf.matmul(c, inputs_hat))

            if i < self.routings - 1:
                b += tf.matmul(outputs, inputs_hat, transpose_b=True)

        return tf.squeeze(outputs)

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):

    output = layers.Conv3D(
        filters=dim_capsule*n_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        name='PrimaryCap_output')(inputs)

    outputs = layers.Reshape(
        target_shape=[-1, dim_capsule],
        name='PrimaryCap_outputs')(output)

    squashed = layers.Lambda(squash, name='PrimaryCap_squash')(outputs)

    return squashed

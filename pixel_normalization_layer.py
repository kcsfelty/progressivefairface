import tensorflow as tf


class PixelNormalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = tf.square(inputs)
        x = tf.reduce_mean(x, axis=1, keepdims=True)
        x = tf.math.rsqrt(x + 1e-8)
        return inputs * x

    def compute_output_shape(self, input_shape):
        return input_shape

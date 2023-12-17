import tensorflow as tf


class Interpolation(tf.keras.layers.Layer):
    """
    Custom Keras layer for linear interpolation between two inputs.

    Parameters:
    - **kwargs: Additional keyword arguments to pass to the superclass.

    Attributes:
    - alpha (tf.Variable): Variable controlling the interpolation weight.

    Methods:
    - call(inputs, *args, **kwargs): Performs the interpolation between two input tensors.
    """
    def __init__(self, **kwargs):
        """
        Initializes the Interpolation layer.

        Parameters:
        - **kwargs: Additional keyword arguments to pass to the superclass.
        """
        super(Interpolation, self).__init__(**kwargs)
        self.alpha = tf.Variable(0., dtype=tf.float64, trainable=False)

    def call(self, inputs, *args, **kwargs):
        """
        Performs linear interpolation between two input tensors.

        Parameters:
        - inputs (tuple): Tuple containing two input tensors to interpolate.
        - *args: Additional positional arguments.
        - **kwargs: Additional keyword arguments.

        Returns:
        - tf.Tensor: Interpolated output tensor.
        """
        assert len(inputs) == 2
        a, b = inputs
        r = interpolation(a, b, self.alpha)
        return r


def interpolation(a, b, t):
    """
    Linear interpolation between two tensors.

    Parameters:
    - a (tf.Tensor): First input tensor.
    - b (tf.Tensor): Second input tensor.
    - t (tf.Tensor): Interpolation weight.

    Returns:
    - tf.Tensor: Interpolated output tensor.
    """
    t = tf.cast(t, tf.float32)
    return a + (b - a) * t

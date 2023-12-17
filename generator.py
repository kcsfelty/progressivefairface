from keras import layers, Model
from keras.constraints import max_norm
from keras.initializers.initializers_v1 import RandomNormal

from interpolation_layer import Interpolation
from pixel_normalization_layer import PixelNormalization


def to_rgb(x):
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)
    x = layers.Conv2D(
        filters=3,
        kernel_size=1,
        padding='same',
        activation='tanh',
        kernel_initializer=init,
        kernel_constraint=const)(x)
    return x


def base_generator(min_resolution_exp=2, latent_dim=64, filters=256, bn=False, kernel=3):
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)

    norm = layers.BatchNormalization if bn else PixelNormalization

    input_layer = layers.Input(shape=(latent_dim,))

    res = 2 ** min_resolution_exp

    x = norm()(input_layer)
    x = layers.Dense(res * res * filters)(x)
    x = layers.Reshape((res, res, filters))(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = norm()(x)
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel,
        padding='same',
        kernel_initializer=init,
        kernel_constraint=const)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = norm()(x)
    x = to_rgb(x)

    return Model(input_layer, x, name="gen_straight_%s" % str(2 ** min_resolution_exp))


def ez_conv(x, filters=256, kernel=3):
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel,
        padding='same',
        kernel_initializer=init,
        kernel_constraint=const)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = PixelNormalization()(x)
    return x


def generator_block(x, filters=256, block_count=2, kernel=3, transpose=False):
    if transpose:
        x = layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel,
            strides=2,
            padding="same",
            kernel_initializer=RandomNormal(stddev=0.02),
            kernel_constraint=max_norm(1.0))(x)
    else:
        x = layers.UpSampling2D()(x)
    for _ in range(block_count - transpose):
        x = ez_conv(x, filters, kernel=kernel)
    return x


def grow_generator(old_generator, filters=256, block_count=2, kernel=3):
    block_end = old_generator.layers[-2].output

    x = generator_block(block_end, filters=filters, block_count=block_count, kernel=kernel)
    out_image_a = to_rgb(x)

    up_sample = layers.UpSampling2D()(block_end)
    out_image_b = old_generator.layers[-1](up_sample)
    c = Interpolation()([out_image_b, out_image_a])

    layer_suffix = str(old_generator.output.shape[0])

    for layer in old_generator.layers:
        layer._name = layer._name + "_" + layer_suffix

    straight = Model(old_generator.input, out_image_a, name="gen_straight_%s" % str(out_image_a.shape[1]))
    mixed = Model(old_generator.input, c, name="gen_mixed_%s" % str(c.shape[1]))

    return straight, mixed

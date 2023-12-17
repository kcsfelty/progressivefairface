from keras import layers, Model
from keras.constraints import max_norm
from keras.initializers.initializers_v1 import RandomNormal

from interpolation_layer import Interpolation


def base_discriminator(min_resolution_exp=2, filters=256, kernel=3):
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)

    input_shape = (2 ** min_resolution_exp, 2 ** min_resolution_exp, 3)
    input_layer = layers.Input(shape=input_shape)
    x = from_rgb(input_layer, filters=filters)
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel,
        padding='same',
        kernel_initializer=init,
        kernel_constraint=const)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(filters)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(1)(x)
    return Model(inputs=input_layer, outputs=x, name="disc_straight_%s" % str(2 ** min_resolution_exp))


def from_rgb(x, filters=256):
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)
    x = layers.Conv2D(
        filters=filters,
        kernel_size=1,
        padding='same',
        kernel_initializer=init,
        kernel_constraint=const)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x


def ez_conv(x, kernel=3, strides=1, filters=256):
    init = RandomNormal(stddev=0.02)
    const = max_norm(1.0)
    x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel,
            padding='same',
            strides=strides,
            kernel_initializer=init,
            kernel_constraint=const)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x


def discriminator_block(x, kernel=3, filters=256, block_count=2):
    x = ez_conv(x, kernel=kernel, strides=2, filters=filters)
    for layer_id in range(block_count - 1):
        x = ez_conv(x, kernel=kernel, filters=filters)
    return x


def grow_discriminator(old_discriminator, filters=256, block_count=2, kernel=3):
    in_shape = list(old_discriminator.input.shape)
    input_shape = (in_shape[-3] * 2, in_shape[-2] * 2, in_shape[-1])
    in_image = layers.Input(shape=input_shape)

    # 'Dumb' scaling
    y = layers.AveragePooling2D()(in_image)
    y = old_discriminator.layers[1](y)
    y = old_discriminator.layers[2](y)

    # 'Smart' scaling
    x = from_rgb(in_image, filters=filters)
    x = discriminator_block(x, filters=filters, block_count=block_count, kernel=kernel)

    c = Interpolation()([y, x])

    for layer in old_discriminator.layers[3:]:
        layer._name = layer._name + "_" + str(input_shape[0] // 2)
        c = layer(x)
        x = layer(x)

    straight = Model(in_image, c, name="disc_straight_%s" % str(input_shape[1]))
    mixed = Model(in_image, x, name="disc_mixed_%s" % str(input_shape[1]))
    return straight, mixed

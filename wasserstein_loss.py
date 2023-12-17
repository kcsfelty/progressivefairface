import tensorflow as tf


def discriminator_loss(real_img, fake_img):
    """
    Calculates the discriminator loss for a pair of real and fake images in a GAN.

    Parameters:
    - real_img (tf.Tensor): Tensor representing the output of the discriminator for real images.
    - fake_img (tf.Tensor): Tensor representing the output of the discriminator for fake images.

    Returns:
    - tf.Tensor: Discriminator loss, computed as the difference between the mean output for fake and real images.
    """
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


def generator_loss(fake_img):
    """
    Calculates the generator loss based on the discriminator's output for fake images in a GAN.

    Parameters:
    - fake_img (tf.Tensor): Tensor representing the output of the discriminator for fake images.

    Returns:
    - tf.Tensor: Generator loss, computed as the negative mean output for fake images.
    """
    return -tf.reduce_mean(fake_img)

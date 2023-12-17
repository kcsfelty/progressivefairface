import datetime
import os

import tensorflow as tf

from config import default_batch_size, default_gradient_count
from dataset import ProgressiveDataset
from discriminator import base_discriminator, grow_discriminator
from generator import base_generator, grow_generator
from progressive_gan import Gan
from visualize_callback import VisualizeCallback
from wasserstein_loss import discriminator_loss, generator_loss


def train(
    data_dir,
    run_name=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),

    # Progressive
    min_resolution_exp=2,
    max_resolution_exp=7,

    # Networks
    filters=256,
    block_count=2,
    g_filters=None,
    d_filters=None,
    latent_dim=64,
    kernel=3,
    n_critic=1,

    # Optimizer
    learning_rate=0.001,
    beta_1=0.0,
    beta_2=0.9,
    epsilon=1e-8,
    reset_opt_stabilize=False,
    reset_opt_transition=False,

    # Output / logging
    mini_epoch_count=32,
    log_path="logs",
    model_path="models",
    image_path="images",
    d_prefix="d_",
    g_prefix="g_",

    # Training
    batch_size=default_batch_size,
    gradient_count=default_gradient_count,

    # Experimental
    lock_old_model_transition=False,
    invert_discriminator_input=False,
    extend_final_stabilize=None,
    measure_g_output=False,

    # Checkpointing
    save_models=True,
    restore_d=True,
    restore_g=True,
    restore_from_run_name=None,
    restore_from_run_exp=2,
    restore_with_stabilize=True,
    restore_opt_warmup=True,
    model_file_extension=".tf",

    # Ignore unknown parameters
    *kwargs,
):

    """
    Trains a Progressive Growing GAN (Generative Adversarial Network) on a given dataset.

    Parameters:
    - data_dir (str): Directory path containing the training data.
    - run_name (str, optional): Name for the current run, default is timestamp-based.

    # Progressive
    - min_resolution_exp (int, optional): Minimum resolution exponent for the dataset, default is 2.
    - max_resolution_exp (int, optional): Maximum resolution exponent for the dataset, default is 7.

    # Networks
    - filters (int, optional): Number of filters in the base generator and discriminator, default is 256.
    - block_count (int, optional): Number of blocks in the base generator and discriminator, default is 2.
    - g_filters (int, optional): Number of filters in the generator, default is the same as 'filters'.
    - d_filters (int, optional): Number of filters in the discriminator, default is the same as 'filters'.
    - latent_dim (int, optional): Dimensionality of the latent space, default is 64.
    - kernel (int, optional): Size of the convolutional kernel, default is 3.
    - n_critic (int, optional): Number of critic iterations per generator iteration, default is 1.

    # Optimizer
    - learning_rate (float, optional): Learning rate for the Adam optimizer, default is 0.001.
    - beta_1 (float, optional): Exponential decay rate for the first moment estimates in Adam, default is 0.0.
    - beta_2 (float, optional): Exponential decay rate for the second moment estimates in Adam, default is 0.9.
    - epsilon (float, optional): Small constant for numerical stability in Adam, default is 1e-8.
    - reset_opt_stabilize (bool, optional): Whether to reset optimizer for stabilization phase, default is False.
    - reset_opt_transition (bool, optional): Whether to reset optimizer for transition phase, default is False.

    # Output / logging
    - mini_epoch_count (int, optional): Number of mini-epochs per epoch, default is 32.
    - log_path (str, optional): Directory path for log files, default is "logs".
    - model_path (str, optional): Directory path to save model files, default is "models".
    - image_path (str, optional): Directory path to save generated images, default is "images".
    - d_prefix (str, optional): Prefix for discriminator model files, default is "d_".
    - g_prefix (str, optional): Prefix for generator model files, default is "g_".

    # Training
    - batch_size (int, optional): Batch size for training, default is the default_batch_size.
    - gradient_count (int, optional): Number of gradients to accumulate before updating weights, default is default_gradient_count.

    # Experimental
    - lock_old_model_transition (bool, optional): Whether to lock models during the transition, default is False.
    - invert_discriminator_input (bool, optional): Whether to invert discriminator input during training, default is False.
    - extend_final_stabilize (int, optional): Number of additional mini-epochs to extend the final stabilization phase, default is None.
    - measure_g_output (bool, optional): Whether to measure and log generator output during training, default is False.

    # Checkpointing
    - save_models (bool, optional): Whether to save trained models, default is True.
    - restore_d (bool, optional): Whether to restore the discriminator from a previous run, default is True.
    - restore_g (bool, optional): Whether to restore the generator from a previous run, default is True.
    - restore_from_run_name (str, optional): Name of the run to restore models from, default is None.
    - restore_from_run_exp (int, optional): Resolution exponent to restore models from, default is 2.
    - restore_with_stabilize (bool, optional): Whether to restore models with the stabilization phase, default is True.
    - restore_opt_warmup (bool, optional): Whether to warm up optimizers during restoration, default is True.
    - model_file_extension (str, optional): File extension for saved model files, default is ".tf".

    Returns:
    - str: Name of the current run.
    """

    g_filters = g_filters or filters
    d_filters = d_filters or filters

    train_counter = tf.Variable(0, dtype=tf.int64, trainable=False)
    log_dir = os.path.join(log_path, run_name)
    file_writer = tf.summary.create_file_writer(log_dir)
    file_writer.set_as_default()
    tf.summary.experimental.set_step(tf.cast(train_counter, dtype=tf.int64))
    os.mkdir(os.path.join(log_dir, image_path))
    d_name = d_prefix + str(restore_from_run_exp) + model_file_extension
    g_name = g_prefix + str(restore_from_run_exp) + model_file_extension

    model_save_path = os.path.join(log_dir, model_path)
    os.mkdir(model_save_path)

    alpha = tf.Variable(0., dtype=tf.float32, trainable=False)

    pd = ProgressiveDataset(
        file_dir=data_dir,
        alpha=alpha,
        min_resolution_exp=min_resolution_exp,
        max_resolution_exp=max_resolution_exp)

    d = None
    g = None

    d_opt = tf.keras.optimizers.Adam(learning_rate, beta_1, beta_2, epsilon)
    g_opt = tf.keras.optimizers.Adam(learning_rate, beta_1, beta_2, epsilon)

    if restore_from_run_name:
        print("Restoring from run:", restore_from_run_name)
        restore_dir = os.path.join(log_path, restore_from_run_name, model_path)

        if restore_d:
            d_restore_path = os.path.join(restore_dir, d_name)
            d = tf.keras.models.load_model(d_restore_path)

        if restore_g:
            g_restore_path = os.path.join(restore_dir, g_name)
            g = tf.keras.models.load_model(g_restore_path)

        if restore_opt_warmup:
            d.trainable = False
            g.trainable = False

    d = d or base_discriminator(
        kernel=kernel,
        min_resolution_exp=min_resolution_exp,
        filters=d_filters)

    g = g or base_generator(
        kernel=kernel,
        min_resolution_exp=min_resolution_exp,
        latent_dim=latent_dim,
        filters=g_filters)

    vc = VisualizeCallback(
        dataset=pd.dataset(min_resolution_exp),
        gen=g,
        latent_dim=latent_dim,
        log_dir=log_dir)

    for res in range(max(min_resolution_exp, restore_from_run_exp or 0), max_resolution_exp + 1):
        data = pd.dataset(res).batch(batch_size[res]).prefetch(mini_epoch_count).repeat()

        vc = VisualizeCallback(
            gen=g,
            latent_dim=latent_dim,
            log_dir=log_dir,
            offset=vc.image_count,
            example_noise=vc.example_noise,
            dataset=data)

        gan = Gan(d, g,
                  alpha=alpha,
                  train_counter=train_counter,
                  latent_dim=latent_dim,
                  n_critic=n_critic,
                  invert_discriminator_input=invert_discriminator_input,
                  measure_g_output=measure_g_output)

        if reset_opt_stabilize:
            d_opt = tf.keras.optimizers.Adam(learning_rate, beta_1, beta_2, epsilon)
            g_opt = tf.keras.optimizers.Adam(learning_rate, beta_1, beta_2, epsilon)

        gan.compile(
            d_opt=d_opt,
            g_opt=g_opt,
            d_loss_fn=discriminator_loss,
            g_loss_fn=generator_loss,
            steps_per_execution=mini_epoch_count,
        )

        if not (res == restore_from_run_exp and not restore_with_stabilize):
            gan(vc.example_noise)
            gan.summary()
            print("Stabilizing", d.input_shape[1:-1])
            gan.fit(data, callbacks=[vc], steps_per_epoch=gradient_count[res])

        if res == max_resolution_exp:
            if extend_final_stabilize:
                print("Extending training for final model", d.input_shape[1:-1])

                gan.compile(
                    d_opt=d_opt,
                    g_opt=g_opt,
                    d_loss_fn=discriminator_loss,
                    g_loss_fn=generator_loss,
                    steps_per_execution=mini_epoch_count,
                )
                gan.fit(data, callbacks=[vc], steps_per_epoch=gradient_count[res] * extend_final_stabilize)
            return

        alpha.assign(0.)

        m_data = pd.dataset(res, interp=True).batch(batch_size[res + 1]).prefetch(mini_epoch_count).repeat()

        if save_models:
            d.save(os.path.join(model_save_path, d_name))
            g.save(os.path.join(model_save_path, g_name))

        if lock_old_model_transition:
            d.trainable = False
            g.trainable = False

        d, m_d = grow_discriminator(
            d,
            filters=d_filters,
            block_count=block_count,
            kernel=kernel)

        g, m_g = grow_generator(
            g,
            filters=g_filters,
            block_count=block_count,
            kernel=kernel)

        m_gan = Gan(m_d, m_g,
                    alpha=alpha,
                    alpha_inc=1 / gradient_count[res + 1],
                    train_counter=train_counter,
                    latent_dim=latent_dim,
                    n_critic=n_critic,
                    invert_discriminator_input=invert_discriminator_input,
                    measure_g_output=measure_g_output)

        if reset_opt_transition:
            d_opt = tf.keras.optimizers.Adam(learning_rate, beta_1, beta_2, epsilon)
            g_opt = tf.keras.optimizers.Adam(learning_rate, beta_1, beta_2, epsilon)

        if restore_from_run_name and restore_opt_warmup:
            d.trainable = True
            g.trainable = True

        m_gan.compile(
            d_opt=d_opt,
            g_opt=g_opt,
            d_loss_fn=discriminator_loss,
            g_loss_fn=generator_loss,
            steps_per_execution=mini_epoch_count)

        vc = VisualizeCallback(
            gen=m_g,
            latent_dim=latent_dim,
            log_dir=log_dir,
            offset=vc.image_count,
            example_noise=vc.example_noise,
            dataset=m_data)

        m_gan(vc.example_noise)
        m_gan.summary()

        print("Transitioning to", d.input_shape[1:-1])
        m_gan.fit(m_data, callbacks=[vc], steps_per_epoch=gradient_count[res + 1])

    return run_name


if __name__ == '__main__':
    train(
        # data_dir="./datasets",
        # data_dir="./datasets/fair_face",
        # data_dir="./datasets/ffhq",
        data_dir="./datasets/celeba-hq",

        max_resolution_exp=6,

        filters=256,
        n_critic=3,
        kernel=3,

        learning_rate=0.001,
        beta_1=0.,
        beta_2=0.9,

        gradient_count={
            2: 2 ** 11,
            3: 2 ** 11,
            4: 2 ** 11,
            5: 2 ** 11,
            6: 2 ** 11,
            7: 2 ** 11},
        mini_epoch_count=2 ** 4,

        extend_final_stabilize=4,
        )

import tensorflow as tf

from interpolation_layer import Interpolation


class Gan(tf.keras.Model):
    """
    Progressive Growing GAN (Generative Adversarial Network) model class.

    Parameters:
    - d (tf.keras.Model): Discriminator model.
    - g (tf.keras.Model): Generator model.
    - alpha (tf.Variable): Variable controlling the interpolation between resolutions.
    - n_critic (int, optional): Number of critic iterations per generator iteration, default is 5.
    - alpha_inc (float, optional): Incremental value for alpha during training, default is 0.
    - train_counter (tf.Variable, optional): Counter for the number of training steps, default is None.
    - latent_dim (int, optional): Dimensionality of the latent space, default is 128.
    - invert_discriminator_input (bool, optional): Whether to invert discriminator input during training, default is False.
    - measure_g_output (bool, optional): Whether to measure and log generator output during training, default is False.
    - w_lambda (float, optional): Weight for the gradient penalty term, default is 10.

    Methods:
    - set_alpha(alpha): Sets the alpha value for interpolation in all interpolation layers of the discriminator and generator.
    - call(x, training=None, mask=None): Forward pass through the generator and discriminator.
    - compile(d_opt, g_opt, d_loss_fn, g_loss_fn, **kwargs): Configures the model for training with specified optimizers and loss functions.
    - train_d(real_images): Trains the discriminator with a batch of real images.
    - train_g(real_images): Trains the generator with a batch of real images.
    - train_step(x, training=False, mask=None): Performs a single training step, updating the discriminator and generator.
    - gradient_penalty(batch_size, real_images, fake_images): Computes the gradient penalty term for regularization.

    Attributes:
    - d_opt (tf.keras.optimizers.Optimizer): Optimizer for the discriminator.
    - g_opt (tf.keras.optimizers.Optimizer): Optimizer for the generator.
    - d_loss_fn (function): Discriminator loss function.
    - g_loss_fn (function): Generator loss function.
    - d_loss_tracker (tf.keras.metrics.Mean): Tracker for discriminator loss.
    - g_loss_tracker (tf.keras.metrics.Mean): Tracker for generator loss.
    - alpha (tf.Variable): Variable controlling the interpolation between resolutions.
    - train_counter (tf.Variable): Counter for the number of training steps.

    Example:
    ```python
    # Instantiate models and create GAN
    discriminator_model = create_discriminator_model()
    generator_model = create_generator_model()
    alpha_variable = tf.Variable(0., dtype=tf.float32, trainable=False)
    gan_model = Gan(d=discriminator_model, g=generator_model, alpha=alpha_variable)

    # Compile the GAN
    gan_model.compile(
        d_opt=tf.keras.optimizers.Adam(learning_rate=0.001),
        g_opt=tf.keras.optimizers.Adam(learning_rate=0.001),
        d_loss_fn=discriminator_loss,
        g_loss_fn=generator_loss
    )

    # Train the GAN
    gan_model.fit(training_data, epochs=100)
    ```
    """
    def __init__(self, d, g, alpha,
                 n_critic=5,
                 alpha_inc=0,
                 train_counter=None,
                 latent_dim=128,
                 invert_discriminator_input=False,
                 measure_g_output=False,
                 w_lambda=10.):
        super(Gan, self).__init__()
        self.d = d
        self.g = g
        self.d_opt = None
        self.g_opt = None
        self.d_loss_fn = None
        self.g_loss_fn = None
        self.d_loss_tracker = tf.keras.metrics.Mean(name='d_loss')
        self.g_loss_tracker = tf.keras.metrics.Mean(name='g_loss')
        self.invert_discriminator_input = invert_discriminator_input
        self.measure_g_output = measure_g_output

        self.n_critic = n_critic
        self.latent_dim = latent_dim
        self.w_lambda = w_lambda
        self.alpha_inc = tf.constant(tf.cast(alpha_inc, dtype=tf.float32), dtype=tf.float32)

        self.train_counter = train_counter
        if self.train_counter is None:
            self.train_counter = tf.Variable(0, dtype=tf.int64, trainable=False)

        self.alpha = alpha
        if self.alpha is None:
            self.alpha = tf.Variable(0., dtype=tf.float32, trainable=False)

        self.set_alpha(self.alpha)

    def set_alpha(self, alpha):
        """
        Sets the alpha value for interpolation in all interpolation layers of the discriminator and generator.

        Parameters:
        - alpha (tf.Variable): Alpha value for interpolation.
        """
        for d_layer in self.d.layers:
            if isinstance(d_layer, Interpolation):
                d_layer.alpha = alpha
        for g_layer in self.g.layers:
            if isinstance(g_layer, Interpolation):
                g_layer.alpha = alpha

    def call(self, x, training=None, mask=None):
        """
        Forward pass through the generator and discriminator.

        Parameters:
        - x (tf.Tensor): Input tensor.
        - training (bool, optional): Indicates whether the model is in training mode, default is None.
        - mask: A mask or list of masks. Defaults to None.

        Returns:
        - tf.Tensor: Output tensor.
        """
        x = self.g(x, training, mask)
        x = self.d(x, training, mask)
        return x

    def compile(self, d_opt, g_opt, d_loss_fn, g_loss_fn, **kwargs):
        """
        Configures the model for training with specified optimizers and loss functions.

        Parameters:
        - d_opt (tf.keras.optimizers.Optimizer): Optimizer for the discriminator.
        - g_opt (tf.keras.optimizers.Optimizer): Optimizer for the generator.
        - d_loss_fn (function): Discriminator loss function.
        - g_loss_fn (function): Generator loss function.
        - **kwargs: Additional arguments passed to the superclass method.
        """
        super().compile(**kwargs)
        self.d_opt = d_opt
        self.g_opt = g_opt
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def train_d(self, real_images):
        """
        Trains the discriminator with a batch of real images.

        Parameters:
        - real_images (tf.Tensor): Batch of real images.
        """
        batch_size = tf.shape(real_images)[0]

        for i in range(self.n_critic):
            noise = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_images = self.g(noise, training=False)
                if self.measure_g_output:
                    tf.summary.scalar("gradient_penalty", tf.reduce_mean(fake_images))

                if self.invert_discriminator_input:
                    real_images *= -1
                    fake_images *= -1

                d_y_hat_fake = self.d(fake_images, training=True)
                d_y_hat_real = self.d(real_images, training=True)

                d_cost = self.d_loss_fn(real_img=d_y_hat_real, fake_img=d_y_hat_fake)
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                tf.summary.scalar("gradient_penalty", gp)
                d_loss = d_cost + gp * self.w_lambda

            d_gradient = tape.gradient(d_loss, self.d.trainable_variables)
            self.d_opt.apply_gradients(
                zip(d_gradient, self.d.trainable_variables))

            self.d_loss_tracker.update_state(d_loss)
            self.train_counter.assign_add(1)
        tf.summary.scalar("Loss/d_loss", self.d_loss_tracker.result())

    def train_g(self, real_images):
        """
        Trains the generator with a batch of real images.

        Parameters:
        - real_images (tf.Tensor): Batch of real images.
        """
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_images = self.g(noise, training=True)
            g_y_hat = self.d(generated_images, training=False)
            g_loss = self.g_loss_fn(g_y_hat)

        gen_gradient = tape.gradient(g_loss, self.g.trainable_variables)
        self.g_opt.apply_gradients(zip(gen_gradient, self.g.trainable_variables))
        self.g_loss_tracker.update_state(g_loss)
        tf.summary.scalar("Loss/g_loss", self.g_loss_tracker.result())

    def train_step(self, x, training=False, mask=None):
        """
        Performs a single training step, updating the discriminator and generator.

        Parameters:
        - x (tf.Tensor): Input tensor.
        - training (bool, optional): Indicates whether the model is in training mode, default is False.
        - mask: A mask or list of masks. Defaults to None.

        Returns:
        - dict: Dictionary containing discriminator and generator losses.
        """
        if training:
            x = tf.image.random_flip_left_right(x)

        self.train_d(x)
        self.train_g(x)
        self.alpha.assign_add(self.alpha_inc)

        result_dict = {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
        }

        self.d_loss_tracker.reset_state()
        self.g_loss_tracker.reset_state()

        return result_dict

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """
        Computes the gradient penalty term for regularization.

        Parameters:
        - batch_size (int): Batch size.
        - real_images (tf.Tensor): Batch of real images.
        - fake_images (tf.Tensor): Batch of fake images.

        Returns:
        - tf.Tensor: Gradient penalty term.
        """
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.d(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

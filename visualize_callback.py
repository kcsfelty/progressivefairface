import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt


class VisualizeCallback(tf.keras.callbacks.Callback):
    def __init__(self, gen, log_dir, example_count=4, latent_dim=512, offset=0, example_noise=None, dataset=None):
        super().__init__()
        self.example_count = example_count
        self.latent_dim = latent_dim
        self.gen = gen
        self.example_noise = example_noise
        if self.example_noise is None:
            self.example_noise = tf.random.normal((example_count ** 2, self.latent_dim))

        self.log_dir = log_dir
        self.image_count = offset

        self.dataset = dataset

    def on_batch_end(self, epoch, logs=None):
        self.visualize()

    def visualize(self):
        test_data = self.dataset.take(1)
        test_data = test_data.as_numpy_iterator()
        test_data = list(test_data)
        test_data = tf.convert_to_tensor(test_data)[0, :self.example_count * self.example_count]
        _, height, width, channels = test_data.shape
        test_data = np.reshape(test_data, (self.example_count, self.example_count, height, width, channels))
        test_data = np.concatenate(np.concatenate(test_data, 1), 1)
        test_data = (test_data + 1) / 2
        test_data = tf.image.resize(test_data, (512, 512), method="nearest")

        out = self.gen(self.example_noise, training=False)

        out += 1
        out /= 2
        out = out.numpy()

        height, width, channels = out.shape[-3:]
        out = np.reshape(out, (self.example_count, self.example_count, height, width, channels))
        out = np.concatenate(np.concatenate(out, 1), 1)
        out = tf.image.resize(out, (512, 512), method="nearest")
        space = np.zeros((512, 32, 3))
        out = np.concatenate([test_data, space, out], 1)
        out = tf.image.resize(out, (512, 1024), method="nearest")

        out = np.array(out * 255)
        out = out.astype(np.uint8)

        filename1 = '%s.png' % (str(int(self.image_count)).zfill(9))
        save_path = os.path.join(self.log_dir, "images", filename1)

        im = Image.fromarray(out)
        im.save(save_path)

        self.image_count += 1


def image_grid(x, size=4):
    x = tf.squeeze(x)
    t = tf.unstack(x[:size * size], num=size*size, axis=0)
    rows = [tf.concat(t[i*size:(i+1)*size], axis=0)
            for i in range(size)]
    image = tf.concat(rows, axis=1)
    return image[None]

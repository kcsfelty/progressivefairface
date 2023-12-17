import tensorflow as tf

from interpolation_layer import interpolation


class ProgressiveDataset:
    def __init__(self, file_dir, alpha, min_resolution_exp=2, max_resolution_exp=7):
        self.file_dir = file_dir
        self.min_resolution_exp = min_resolution_exp
        self.max_resolution_exp = max_resolution_exp
        self.current_resolution_exp = min_resolution_exp
        self.alpha = alpha

        @tf.function
        def map_space(x):
            return tf.cast(x / 255 * 2 - 1, dtype=tf.float32)

        @tf.function
        def map_interp(x, y):
            shape = y.shape[:-1]
            x = tf.image.resize(x, shape, method='nearest')
            x = interpolation(x, y, alpha)
            return x

        dataset_hash = {}
        for res in range(min_resolution_exp, max_resolution_exp + 1):
            dataset = tf.keras.utils.image_dataset_from_directory(
                directory=file_dir,
                image_size=(2 ** res, 2 ** res),
                batch_size=None,
                label_mode=None,
                subset=None,
                shuffle=False,
                interpolation='nearest')
            dataset = dataset.map(map_space)
            dataset_hash[res] = dataset

        m_dataset_hash = {}
        for res in range(min_resolution_exp, max_resolution_exp):
            dataset = tf.data.Dataset.zip((dataset_hash[res], dataset_hash[res + 1]))
            dataset = dataset.map(map_interp)
            m_dataset_hash[res] = dataset

        self.dataset_hash = dataset_hash
        self.m_dataset_hash = m_dataset_hash

    def dataset(self, res, interp=False):
        if interp:
            return self.m_dataset_hash[res]
        return self.dataset_hash[res]

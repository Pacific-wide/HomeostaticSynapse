import numpy as np
import tensorflow as tf

def load_mnist_datasets():
    train, test = tf.keras.datasets.mnist.load_data()
    x_train, y_train = train
    x_test, y_test = test
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    return dataset_train, dataset_test
import numpy as np
import tensorflow as tf


class Dataset(object):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels


class Mnist(object):
    def __init__(self):
        self.load()
        self.normalize()

    def load(self):
        train, test = tf.keras.datasets.mnist.load_data()
        x_train, y_train = train
        x_test, y_test = test
        self.train = Dataset(x_train, y_train)
        self.test = Dataset(x_test, y_test)

    def normalize(self):
        x_train = self.train.features.astype(np.float32)  # (60000, 28, 28)
        x_test = self.test.features.astype(np.float32)  # (10000, 28, 28)
        self.train.features = x_train.reshape(x_train.shape[0], -1) / 255.0  # (60000, 784)
        self.test.features = x_test.reshape(x_test.shape[0], -1) / 255.0  # (10000, 784)


class PermMnist(Mnist):
    def __init__(self, permutation):
        super(PermMnist, self).__init__()
        self.permutation = permutation
        self.permute()

    def permute(self):
        self.train.features = self.train.features[self.permutation]


class RandPermMnist(PermMnist):
    def __init__(self):
        permutation = np.random.permutation(784)
        super(RandPermMnist, self).__init__(permutation)


class SetOfRandPermMnist(object):
    def __init__(self, n_task):
        self.list = []
        self.n_task = n_task
        self.generate()

    def generate(self):
        for i in range(self.n_task):
            self.list.append(RandPermMnist())

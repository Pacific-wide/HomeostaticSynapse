import numpy as np
import tensorflow as tf


class Mnist(object):
    def __init__(self):
        self.load()
        self.normalize()

    def load(self):
        train, test = tf.keras.datasets.mnist.load_data()
        self.x_train, self.y_train = train
        self.x_test, self.y_test = test

    def normalize(self):
        self.x_train = self.x_train.astype(np.float32)  # (60000, 28, 28)
        self.x_test = self.x_test.astype(np.float32)  # (10000, 28, 28)
        self.x_train = self.x_train.reshape(self.x_train.shape[0], -1) / 255.0  # (60000, 784)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1) / 255.0  # (10000, 784)

        self.y_train = self.y_train.astype(np.int64)  # (60000, )
        self.y_test = self.y_test.astype(np.int64)  # (10000, )


class PermMnist(Mnist):
    def __init__(self, permutation):
        super(PermMnist, self).__init__()
        self.permutation = permutation
        self.permute()

    def permute(self):
        self.x_train = self.x_train[:, self.permutation]
        self.x_test = self.x_test[:, self.permutation]


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

    def concat(self):
        x_train_list = []
        y_train_list = []
        for i in range(self.n_task):
            x_train_list.append(self.list[i].x_train)
            y_train_list.append(self.list[i].y_train)

        multi_dataset = RandPermMnist()
        multi_dataset.x_train = np.concatenate(x_train_list, axis=0)
        multi_dataset.y_train = np.concatenate(y_train_list, axis=0)

        return multi_dataset

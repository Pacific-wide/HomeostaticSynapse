import numpy as np
import tensorflow as tf
import scipy.io
from PIL import Image


class DataSet(object):
    def __init__(self):
        self.load()
        self.normalize()
        self.d_in = 0

    def load(self):
        pass

    def normalize(self):
        pass


class Mnist(DataSet):
    def __init__(self):
        super(Mnist, self).__init__()
        self.d_in = 28 * 28

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


class RotaMnist(Mnist):
    def __init__(self, angle):
        super(RotaMnist, self).__init__()
        self.angle = angle
        self.x_train = self.rotate(self.x_train)
        self.x_test = self.rotate(self.x_test)

    def rotate(self, data):
        data = data.reshape(-1, 28, 28)
        shape = data.shape
        result = np.zeros(shape)

        for i in range(shape[0]):
            img = Image.fromarray(data[i], mode='L')
            result[i] = img.rotate(self.angle)

        result = result.reshape(-1, 28*28)
        result = result.astype(np.float32)

        return result


class RandRotaMnist(RotaMnist):
    def __init__(self):
        angle = np.random.randint(360)
        super(RandRotaMnist, self).__init__(angle)


class SetOfRandRotaMnist(object):
    def __init__(self, n_task):
        self.list = []
        self.n_task = n_task
        self.generate()

    def generate(self):
        for i in range(self.n_task):
            self.list.append(RandRotaMnist())

    def concat(self):
        x_train_list = []
        y_train_list = []
        for i in range(self.n_task):
            x_train_list.append(self.list[i].x_train)
            y_train_list.append(self.list[i].y_train)

        multi_dataset = RandRotaMnist()
        multi_dataset.x_train = np.concatenate(x_train_list, axis=0)
        multi_dataset.y_train = np.concatenate(y_train_list, axis=0)

        return multi_dataset


class SVHN(DataSet):
    def __init__(self):
        super(SVHN, self).__init__()
        self.d_in = 32 * 32 * 3

    def load(self):
        train = scipy.io.loadmat('dataset/train.mat')
        test = scipy.io.loadmat('dataset/test.mat')

        self.x_train = train['X']   # (32, 32, 3, 73257)
        self.y_train = train['y']   # (73257, 1)
        self.x_test = test['X']     # (32, 32, 3, 26032)
        self.y_test = test['y']     # (26032, 1)


    def normalize(self):
        self.x_train = self.x_train.astype(np.float32)
        self.x_test = self.x_test.astype(np.float32)

        self.x_train = self.x_train.reshape(32*32*3, -1) / 255.0  # (60000, 784)
        self.x_test = self.x_test.reshape(32*32*3, -1) / 255.0  # (10000, 784)

        self.x_train = np.swapaxes(self.x_train, 0, 1)
        self.x_test = np.swapaxes(self.x_test, 0, 1)

        self.y_train = self.y_train.astype(np.int64)
        self.y_test = self.y_test.astype(np.int64)
        self.y_train = self.y_train.reshape(-1)   # (73257, )
        self.y_test = self.y_test.reshape(-1)   # (26032, )




class CIFAR10(DataSet):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.d_in = 32 * 32 * 3

    def load(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()



class FashionMnist(DataSet):
    def __init__(self):
        super(FashionMnist, self).__init__()
        self.d_in = 32 * 32 * 3

    def load(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()

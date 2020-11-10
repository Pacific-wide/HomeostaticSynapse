import numpy as np
import tensorflow as tf
from PIL import Image


class DataSet(object):
    def __init__(self):
        self.load()
        self.d_in = 0
        self.normalize()
        self.n_train = 0
        self.n_test = 0

    def load(self):
        pass

    def normalize(self):
        pass

    def set_data(self, tuple):
        self.x_train = tuple[0]
        self.y_train = tuple[1]


class MNIST(DataSet):
    def __init__(self):
        super(MNIST, self).__init__()
        self.row = 28
        self.d_in = self.row * self.row
        self.n_train = self.x_train.shape[0]
        self.n_test = self.x_test.shape[0]

    def load(self):
        train, test = tf.keras.datasets.mnist.load_data()
        self.x_train, self.y_train = train
        self.x_test, self.y_test = test

    def flatten(self):
        self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)   # (60000, 784)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)   # (10000, 784)

    def normalize(self):
        self.x_train = self.x_train.astype(np.float32) / 255.0   # (60000, 28, 28)
        self.x_test = self.x_test.astype(np.float32) / 255.0   # (10000, 28, 28)

        self.y_train = self.y_train.astype(np.int64)  # (60000, )
        self.y_test = self.y_test.astype(np.int64)  # (10000, )

    def reshape3D(self):
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.row, self.row, 1)  # (60000, 28, 28 ,1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.row, self.row, 1)  # (10000, 28, 28 ,1)


class MNISTPERM(MNIST):
    def __init__(self, perm):
        super(MNISTPERM, self).__init__()
        self.perm = perm
        self.permute()
        self.label_perm = np.random.permutation(10)

    def permute(self):
        self.flatten()
        self.x_train = self.x_train[:, self.perm]
        self.x_test = self.x_test[:, self.perm]

    def permute_label(self):
        for i, label in enumerate(self.y_train):
            self.y_train[i] = self.label_perm[label]

        for j, label in enumerate(self.y_test):
            self.y_test[j] = self.label_perm[label]


class RandMNISTPERM(MNISTPERM):
    def __init__(self):
        pixels = 28 * 28
        perm = np.random.permutation(pixels)
        super(RandMNISTPERM, self).__init__(perm)


class RowMNISTPERM(MNISTPERM):
    def __init__(self, perm):
        super(RowMNISTPERM, self).__init__(perm)

    def permute(self):
        self.x_train = self.x_train[:, self.perm, :]
        self.x_test = self.x_test[:, self.perm, :]

        self.flatten()


class RandRowMNISTPERM(RowMNISTPERM):
    def __init__(self):
        row = 28
        perm = np.random.permutation(row)
        super(RandRowMNISTPERM, self).__init__(perm)


class ColMNISTPERM(MNISTPERM):
    def __init__(self, perm):
        super(ColMNISTPERM, self).__init__(perm)

    def permute(self):
        self.x_train = self.x_train[:, :, self.perm]
        self.x_test = self.x_test[:, :, self.perm]

        self.flatten()


class RandColMNISTPERM(ColMNISTPERM):
    def __init__(self):
        row = 28
        perm = np.random.permutation(row)
        super(RandColMNISTPERM, self).__init__(perm)


class MNISTBPERM(MNISTPERM):
    def __init__(self, perm, n_grid):
        self.n_grid = n_grid
        self.row = 28
        self.col = 28
        self.n_block = int(self.row / self.n_grid)

        super(MNISTBPERM, self).__init__(perm)

    def permute(self):
        for i, train_sample in enumerate(self.x_train):
            tr_blocks = self.make_blocks(train_sample)
            tr_perm_sample = self.permute_blocks(self.perm, tr_blocks)

            self.x_train[i] = tr_perm_sample

        for i, test_sample in enumerate(self.x_test):
            te_blocks = self.make_blocks(test_sample)
            te_perm_sample = self.permute_blocks(self.perm, te_blocks)

            self.x_test[i] = te_perm_sample

        self.flatten()

    def make_blocks(self, sample):
        x = sample
        blocks = []
        for i in range(self.n_grid):
            for j in range(self.n_grid):
                n = self.n_block
                sub_x = x[n*i:n*(i+1), n*j:n*(j+1)]
                blocks.append(sub_x)

        return blocks

    def permute_blocks(self, perm, blocks):
        perm_x = np.zeros((self.row, self.col), dtype=float)
        for index, order in enumerate(perm):
            i = int(order / self.n_grid)
            j = order % self.n_grid
            n = self.n_block

            perm_x[n*i:n*(i+1), n*j:n*(j+1)] = blocks[index]

        return perm_x


class RandMNISTBPERM(MNISTBPERM):
    def __init__(self, n_grid):
        self.grid_pixels = n_grid * n_grid
        perm = np.random.permutation(self.grid_pixels)
        super(RandMNISTBPERM, self).__init__(perm, n_grid)


class MNISTROTA(MNIST):
    def __init__(self, angle):
        super(MNISTROTA, self).__init__()
        self.angle = angle
        self.x_train = self.rotate(self.x_train)
        self.x_test = self.rotate(self.x_test)

    def rotate(self, data):
        data = data.reshape(-1, self.row, self.row)
        shape = data.shape
        result = np.zeros(shape)

        for i in range(shape[0]):
            img = Image.fromarray(data[i], mode='L')
            result[i] = img.rotate(self.angle)

        result = result.reshape(-1, self.d_in)
        result = result.astype(np.float32)

        return result


class RandMNISTROTA(MNISTROTA):
    def __init__(self):
        angle = np.random.randint(360)
        print("Task " + ":" + str(angle))
        super(RandMNISTROTA, self).__init__(angle)


class MNISTSPLIT(MNIST):
    def __init__(self, label_list):
        super(MNISTSPLIT, self).__init__()
        self.label_list = label_list
        self.split()
        self.flatten()

    def split(self):
        train_index = self.obtain_index__from_label(self.y_train)
        test_index = self.obtain_index__from_label(self.y_test)

        self.x_train = self.x_train[train_index]
        self.x_test = self.x_test[test_index]

        self.y_train = self.y_train[train_index]
        self.y_test = self.y_test[test_index]

    def obtain_index__from_label(self, y_tensor):
        index_list = []
        for label in self.label_list:
            indexes = np.where(label == y_tensor)[0]
            index_list.extend(indexes)
            index_list.sort()

        return index_list


class CIFAR10(DataSet):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.row = 32
        self.d_in = self.row * self.row * 3
        self.n_train = self.x_train.shape[0]
        self.n_test = self.x_test.shape[0]

    def load(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()

    def normalize(self):
        self.x_train = self.x_train.astype(np.float32)  # (50000, 32, 32, 3)
        self.x_test = self.x_test.astype(np.float32)  # (10000, 32, 32, 3)
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

        self.y_train = self.y_train.astype(np.int64)
        self.y_test = self.y_test.astype(np.int64)
        self.y_train = self.y_train.reshape(self.y_train.shape[0])
        self.y_test = self.y_test.reshape(self.y_test.shape[0])

    def flatten(self):
        self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)

    def flatten_2D(self):
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 1024, 3)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 1024, 3)

    def reshape3D(self):
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.row, self.row, 3)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.row, self.row, 3)


class CIFAR10ROTA(CIFAR10):
    def __init__(self, angle):
        super(CIFAR10ROTA, self).__init__()
        self.angle = angle
        self.x_train = self.rotate(self.x_train)
        self.x_test = self.rotate(self.x_test)

        self.flatten()

    def rotate(self, data):
        shape = data.shape
        result = np.zeros(shape)

        for i in range(shape[0]):
            for j in range(3):
                map = Image.fromarray(data[i,:,:,j], mode='L')
                rotated_map = map.rotate(self.angle)
                result[i,:,:,j] = np.array(rotated_map)

        result = result.astype(np.float32)

        return result


class RandCIFAR10ROTA(CIFAR10ROTA):
    def __init__(self):
        angle = np.random.randint(360)
        print("Task " + ":" + str(angle))
        super(RandCIFAR10ROTA, self).__init__(angle)


class CIFAR10PERM(CIFAR10):
    def __init__(self, perm):
        super(CIFAR10PERM, self).__init__()
        self.perm = perm
        self.permute()
        self.flatten()

    def permute(self):
        self.flatten_2D()
        self.x_train = self.x_train[:, self.perm, :]
        self.x_test = self.x_test[:, self.perm, :]


class CIFAR10BPERM(CIFAR10PERM):
    def __init__(self, perm, n_grid):
        self.n_grid = n_grid
        self.row = 32
        self.col = 32
        self.n_block = int(self.row / self.n_grid)

        super(CIFAR10BPERM, self).__init__(perm)

    def permute(self):
        for i, train_sample in enumerate(self.x_train):
            tr_blocks = self.make_blocks(train_sample) # (32 ,32, 3)
            tr_perm_sample = self.permute_blocks(self.perm, tr_blocks)

            self.x_train[i] = tr_perm_sample

        for i, test_sample in enumerate(self.x_test):
            te_blocks = self.make_blocks(test_sample)
            te_perm_sample = self.permute_blocks(self.perm, te_blocks)

            self.x_test[i] = te_perm_sample

        self.flatten()

    def make_blocks(self, sample):
        x = sample
        blocks = []
        for i in range(self.n_grid):
            for j in range(self.n_grid):
                n = self.n_block
                sub_x = x[n*i:n*(i+1), n*j:n*(j+1), :]
                blocks.append(sub_x)

        return blocks

    def permute_blocks(self, perm, blocks):
        perm_x = np.zeros((self.row, self.col, 3), dtype=float)
        for index, order in enumerate(perm):
            i = int(order / self.n_grid)
            j = order % self.n_grid
            n = self.n_block

            perm_x[n*i:n*(i+1), n*j:n*(j+1), :] = blocks[index]

        return perm_x


class RandCIFAR10PERM(CIFAR10PERM):
    def __init__(self):
        perm = np.random.permutation(32*32)
        super(RandCIFAR10PERM, self).__init__(perm)


class RandCIFAR10BPERM(CIFAR10BPERM):
    def __init__(self, n_grid):
        self.grid_pixels = n_grid * n_grid
        perm = np.random.permutation(self.grid_pixels)
        super(RandCIFAR10BPERM, self).__init__(perm, n_grid)



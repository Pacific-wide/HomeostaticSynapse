import numpy as np
import tensorflow as tf
import scipy.io
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


class Mnist(DataSet):
    def __init__(self):
        super(Mnist, self).__init__()
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


class SwapMnist(Mnist):
    def __init__(self, perm):
        super(SwapMnist, self).__init__()
        self.perm = perm
        self.in_perm = self.perm
        self.out_perm = np.flip(self.perm)

        self.flatten()
        self.swap_pixel()

    def swap_pixel(self):
        for (i, o) in zip(self.in_perm, self.out_perm):
            temp_pixel = self.x_train[:, i]
            self.x_train[:, i] = self.x_train[:, o]
            self.x_train[:, o] = temp_pixel

            temp_pixel = self.x_test[:, i]
            self.x_test[:, i] = self.x_test[:, o]
            self.x_test[:, o] = temp_pixel


class PermMnist(Mnist):
    def __init__(self, perm):
        super(PermMnist, self).__init__()
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


class RandPermMnist(PermMnist):
    def __init__(self):
        pixels = 28 * 28
        perm = np.random.permutation(pixels)
        super(RandPermMnist, self).__init__(perm)


class RowPermMnist(PermMnist):
    def __init__(self, perm):
        super(RowPermMnist, self).__init__(perm)

    def permute(self):
        self.x_train = self.x_train[:, self.perm, :]
        self.x_test = self.x_test[:, self.perm, :]

        self.flatten()


class RandRowPermMnist(RowPermMnist):
    def __init__(self):
        row = 28
        perm = np.random.permutation(row)
        super(RandRowPermMnist, self).__init__(perm)


class ColPermMnist(PermMnist):
    def __init__(self, perm):
        super(ColPermMnist, self).__init__(perm)

    def permute(self):
        self.x_train = self.x_train[:, :, self.perm]
        self.x_test = self.x_test[:, :, self.perm]

        self.flatten()


class RandColPermMnist(ColPermMnist):
    def __init__(self):
        row = 28
        perm = np.random.permutation(row)
        super(RandColPermMnist, self).__init__(perm)


class WholePermMnist(PermMnist):
    def __init__(self, row_perm, col_perm):
        super(WholePermMnist, self).__init__(0)
        self.row_perm = row_perm
        self.col_perm = col_perm

    def permute(self):
        self.x_train = self.x_train[:, self.row_perm, :]
        self.x_test = self.x_test[:, self.row_perm, :]

        self.x_train = self.x_train[:, :, self.col_perm]
        self.x_test = self.x_test[:, :, self.col_perm]

        self.flatten()


class RandWholePermMnist(WholePermMnist):
    def __init__(self):
        row = 28
        self.row_perm = np.random.permutation(row)
        self.col_perm = np.random.permutation(row)
        super(RandWholePermMnist, self).__init__(self.row_perm, self.col_perm)


class GridPermMnist(PermMnist):
    def __init__(self, perm, n_grid):
        self.n_grid = n_grid
        self.row = 28
        self.col = 28
        self.n_block = int(self.row / self.n_grid)

        super(GridPermMnist, self).__init__(perm)

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


class RandGridPermMnist(GridPermMnist):
    def __init__(self, n_grid):
        self.grid_pixels = n_grid * n_grid
        perm = np.random.permutation(self.grid_pixels)
        print(perm)
        super(RandGridPermMnist, self).__init__(perm, n_grid)


class RotaMnist(Mnist):
    def __init__(self, angle):
        super(RotaMnist, self).__init__()
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


class RandRotaMnist(RotaMnist):
    def __init__(self):
        angle = np.random.randint(360)
        print("Task " + ":" + str(angle))
        super(RandRotaMnist, self).__init__(angle)


class BlockBoxMnist(Mnist):
    def __init__(self, block_ratio, i_row, i_col):
        super(BlockBoxMnist, self).__init__()
        self.block_ratio = block_ratio
        self.i_row = i_row
        self.i_col = i_col
        self.block_box()
        self.flatten()
        print(self.x_train.shape)

    def block_box(self):
        block_length = int(self.block_ratio * self.row)

        self.x_train[:, self.i_row:self.i_row+block_length, self.i_col:self.i_col+block_length] = 0
        self.x_test[:, self.i_row:self.i_row+block_length, self.i_col:self.i_col+block_length] = 0


class RandBlockBoxMnist(BlockBoxMnist):
    def __init__(self, block_ratio):
        row = 28
        i_row = np.random.randint(row)
        i_col = np.random.randint(row)
        super(RandBlockBoxMnist, self).__init__(block_ratio, i_row, i_col)


class SplitMnist(Mnist):
    def __init__(self, label_list):
        super(SplitMnist, self).__init__()
        self.label_list = label_list
        self.split()
        self.flatten()

    def split(self):
        train_index = self.obtain_index__from_label(self.y_train)
        test_index = self.obtain_index__from_label(self.y_test)

        self.x_train = self.x_train[train_index]
        self.x_test = self.x_test[test_index]
        print(self.x_train.shape)
        print(self.x_test.shape)

        self.y_train = self.y_train[train_index]
        self.y_test = self.y_test[test_index]

        print(self.y_train.shape)
        print(self.y_test.shape)

    def obtain_index__from_label(self, y_tensor):
        index_list = []
        for label in self.label_list:
            indexes = np.where(label == y_tensor)[0]
            index_list.extend(indexes)
            index_list.sort()

        return index_list


class SVHN(DataSet):
    def __init__(self):
        super(SVHN, self).__init__()
        self.d_in = 32 * 32 * 3
        self.n_train = self.x_train.shape[0]
        self.n_train = self.x_test.shape[0]

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
        self.row = 32
        self.d_in = self.row * self.row * 3
        self.n_train = self.x_train.shape[0]
        self.n_test = self.x_test.shape[0]

    def load(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()

    def normalize(self):
        self.x_train = self.x_train.astype(np.float32)  # (50000, 32, 32, 3)
        self.x_test = self.x_test.astype(np.float32)  # (10000, 32, 32, 3)
        self.x_train = self.x_train / 255.0 # (50000, 3*1024)
        self.x_test = self.x_test / 255.0 # (10000, 3*1024)

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


class RotaCIFAR10(CIFAR10):
    def __init__(self, angle):
        super(RotaCIFAR10, self).__init__()
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


class RandRotaCIFAR10(RotaCIFAR10):
    def __init__(self):
        angle = np.random.randint(360)
        print("Task " + ":" + str(angle))
        super(RandRotaCIFAR10, self).__init__(angle)


class PermCIFAR10(CIFAR10):
    def __init__(self, perm):
        super(PermCIFAR10, self).__init__()
        self.perm = perm
        self.permute()
        self.flatten()

    def permute(self):
        self.flatten_2D()
        self.x_train = self.x_train[:, self.perm, :]
        self.x_test = self.x_test[:, self.perm, :]


class GridPermCIFAR10(PermCIFAR10):
    def __init__(self, perm, n_grid):
        self.n_grid = n_grid
        self.row = 32
        self.col = 32
        self.n_block = int(self.row / self.n_grid)

        super(GridPermCIFAR10, self).__init__(perm)

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


class RandPermCIFAR10(PermCIFAR10):
    def __init__(self):
        perm = np.random.permutation(32*32)
        super(RandPermCIFAR10, self).__init__(perm)


class RandGridPermCIFAR10(GridPermCIFAR10):
    def __init__(self, n_grid):
        self.grid_pixels = n_grid * n_grid
        perm = np.random.permutation(self.grid_pixels)
        super(RandGridPermCIFAR10, self).__init__(perm, n_grid)


class FashionMnist(DataSet):
    def __init__(self):
        super(FashionMnist, self).__init__()
        self.d_in = 32 * 32 * 3

    def load(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()

import numpy as np
import dataset.dataset as ds


class SetOfDataSet(object):
    def __init__(self, n_task):
        self.list = []
        self.n_task = n_task
        self.generate()

    def generate(self):
        pass

    def concat(self):
        x_train_list = []
        y_train_list = []
        for i in range(self.n_task):
            x_train_list.append(self.list[i].x_train)
            y_train_list.append(self.list[i].y_train)

        multi_dataset = ds.RandPermMnist()
        multi_dataset.x_train = np.concatenate(x_train_list, axis=0)
        multi_dataset.y_train = np.concatenate(y_train_list, axis=0)

        return multi_dataset


class SetOfCurriculumGrid(SetOfDataSet):
    def __init__(self, n_task, n_grid):
        super(SetOfCurriculumGrid, self).__init__(n_task)
        self.n_gradual_task = int((2/3) * self.n_task)
        self.n_perm_task = n_task - self.n_gradual_task

    def generate(self):
        for index in range(self.n_gradual_task):
            self.list.append(ds.GridPermMnist(self.grid_perm, self.n_grid))
            self.swap_perm(index)

        for index in range(self.n_perm_task):
            self.list.append(ds.RandGridPermMnist)


class SetOfSwapGrid(SetOfDataSet):
    def __init__(self, n_task, n_grid, step):
        self.n_grid = n_grid
        self.step = step
        self.grid_blocks = self.n_grid * self.n_grid
        # self.grid_perm = np.arange(self.grid_blocks)
        self.grid_perm = np.random.permutation(self.grid_blocks)
        self.pixel_perm = np.random.permutation(self.grid_blocks)

        super(SetOfSwapGrid, self).__init__(n_task)

    def generate(self):
        for index in range(self.n_task):
            self.list.append(ds.GridPermMnist(self.grid_perm, self.n_grid))
            for j in range(self.step):
                self.swap_perm(index*(j+1))
            print(self.grid_perm)

    def swap_perm(self, index):
        input_index = index % self.grid_blocks
        i = self.pixel_perm[input_index]
        out_index = (3*index+1) % self.grid_blocks
        o = self.pixel_perm[out_index]
        self.grid_perm[i] = o
        self.grid_perm[o] = i


class SetOfMnistPlusGrid(SetOfDataSet):
    def __init__(self, n_task, n_grid):
        self.n_grid = n_grid
        super(SetOfMnistPlusGrid, self).__init__(n_task)

    def generate(self):
        self.list.append(ds.Mnist())
        self.list[0].flatten()
        for i in range(1, self.n_task):
            self.list.append(ds.RandGridPermMnist(self.n_grid))


class SetOfMnistPlusGridCIFAR10(SetOfDataSet):
    def __init__(self, n_task, n_grid):
        self.n_grid = n_grid
        super(SetOfMnistPlusGridCIFAR10, self).__init__(n_task)

    def generate(self):
        self.list.append(ds.CIFAR10())
        self.list[0].flatten()
        for i in range(1, self.n_task):
            self.list.append(ds.RandGridPermCIFAR10(self.n_grid))


class SetOfMnistPlusRota(SetOfDataSet):
    def __init__(self, n_task, angle):
        self.angle = angle
        super(SetOfMnistPlusRota, self).__init__(n_task)

    def generate(self):
        self.list.append(ds.Mnist())
        self.list[0].flatten()
        for i in range(1, self.n_task):
            self.list.append(ds.RotaMnist(self.angle))


class SetOfMnistPlusSwap(SetOfDataSet):
    def __init__(self, n_task, n_swap_epoch):
        n_pixel = 784
        self.perm = np.zeros(n_pixel * n_swap_epoch, dtype=int)
        for i in np.arange(n_swap_epoch):
            self.perm[n_pixel*i:n_pixel*(i+1)] = np.random.permutation(n_pixel)

        super(SetOfMnistPlusSwap, self).__init__(n_task)

    def generate(self):
        self.list.append(ds.Mnist())
        self.list[0].flatten()
        for i in range(1, self.n_task):
            self.list.append(ds.SwapMnist(self.perm))


class SetOfRandPermMnist(SetOfDataSet):
    def __init__(self, n_task):
        super(SetOfRandPermMnist, self).__init__(n_task)

    def generate(self):
        for i in range(self.n_task):
            self.list.append(ds.RandPermMnist())


class SetOfRandRowPermMnist(SetOfDataSet):
    def __init__(self, n_task):
        super(SetOfRandRowPermMnist, self).__init__(n_task)

    def generate(self):
        for i in range(self.n_task):
            self.list.append(ds.RandRowPermMnist())


class SetOfRandColPermMnist(SetOfDataSet):
    def __init__(self, n_task):
        super(SetOfRandColPermMnist, self).__init__(n_task)

    def generate(self):
        for i in range(self.n_task):
            self.list.append(ds.RandColPermMnist())


class SetOfRandWholePermMnist(SetOfDataSet):
    def __init__(self, n_task):
        super(SetOfRandWholePermMnist, self).__init__(n_task)

    def generate(self):
        for i in range(self.n_task):
            self.list.append(ds.RandWholePermMnist())


class SetOfRandGridPermMnist(SetOfDataSet):
    def __init__(self, n_task, n_grid):
        self.n_grid = n_grid
        super(SetOfRandGridPermMnist, self).__init__(n_task)

    def generate(self):
        for i in range(self.n_task):
            self.list.append(ds.RandGridPermMnist(self.n_grid))


class SetOfRandRotaMnist(SetOfDataSet):
    def __init__(self, n_task):
        super(SetOfRandRotaMnist, self).__init__(n_task)

    def generate(self):
        for i in range(self.n_task):
            self.list.append(ds.RandRotaMnist())


class SetOfGradualRotaMnist(SetOfDataSet):
    def __init__(self, n_task, range):
        self.range = range
        super(SetOfGradualRotaMnist, self).__init__(n_task)

    def generate(self):
        for i in range(self.n_task):
            angle = int((i + 1) * 360 * self.range / self.n_task)
            self.list.append(ds.RotaMnist(angle))


class SetOfGradualSplitMnist(SetOfDataSet):
    def __init__(self, n_task):
        self.period = int(10 / n_task)
        super(SetOfGradualSplitMnist, self).__init__(n_task)

    def generate(self):
        for i in range(self.n_task):
            label_list = range(10)[self.period*i:self.period*(i+1)]
            print(label_list)
            self.list.append(ds.SplitMnist(label_list))


class SetOfAlternativeMnist(object):
    def __init__(self, n_task):
        self.list = []
        self.n_task = int(n_task/2)
        self.generate()

    def generate(self):
        for i in range(self.n_task):
            self.list.append(ds.RandPermMnist())
            self.list.append(ds.RandRotaMnist())


class SetOfRandBlockBoxMnist(SetOfDataSet):
    def __init__(self, n_task, ratio):
        self.ratio = ratio
        super(SetOfRandBlockBoxMnist, self).__init__(n_task)

    def generate(self):
        for i in range(self.n_task):
            self.list.append(ds.RandBlockBoxMnist(self.ratio))


class SetOfRandPermCIFAR10(object):
    def __init__(self, n_task):
        self.list = []
        self.n_task = n_task
        self.generate()

    def generate(self):
        for i in range(self.n_task):
            self.list.append(ds.RandPermCIFAR10())

    def concat(self):
        x_train_list = []
        y_train_list = []
        for i in range(self.n_task):
            x_train_list.append(self.list[i].x_train)
            y_train_list.append(self.list[i].y_train)

        multi_dataset = ds.RandPermCIFAR10()
        multi_dataset.x_train = np.concatenate(x_train_list, axis=0)
        multi_dataset.y_train = np.concatenate(y_train_list, axis=0)

        return multi_dataset

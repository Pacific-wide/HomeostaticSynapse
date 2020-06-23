import numpy as np
import dataset.dataset as ds


class SetOfDataSet(object):
    def __init__(self, n_task):
        self.list = []
        self.n_task = n_task
        self.generate()

    def set_list(self, _list):
        self.list = _list

    def generate(self):
        pass

    def concat(self):
        x_train_list = []
        y_train_list = []
        for i in range(self.n_task):
            x_train_list.append(self.list[i].x_train)
            y_train_list.append(self.list[i].y_train)

        multi_dataset = ds.MNIST()
        multi_dataset.x_train = np.concatenate(x_train_list, axis=0)
        multi_dataset.y_train = np.concatenate(y_train_list, axis=0)

        return multi_dataset


class SetOfSwapBlock(SetOfDataSet):
    def __init__(self, n_task, n_grid, step):
        self.n_grid = n_grid
        self.step = step
        self.grid_blocks = self.n_grid * self.n_grid
        # self.grid_perm = np.arange(self.grid_blocks)
        self.grid_perm = np.random.permutation(self.grid_blocks)
        self.pixel_perm = np.random.permutation(self.grid_blocks)

        super(SetOfSwapBlock, self).__init__(n_task)

    def generate(self):
        for index in range(self.n_task):
            self.list.append(ds.MNISTBPERM(self.grid_perm, self.n_grid))
            for j in range(self.step):
                self.swap_perm(index*(j+1))

    def swap_perm(self, index):
        input_index = index % self.grid_blocks
        i = self.pixel_perm[input_index]
        out_index = (3*index+1) % self.grid_blocks
        o = self.pixel_perm[out_index]
        self.grid_perm[i] = o
        self.grid_perm[o] = i


class SetOfMNISTPlusMNISTBPERM(SetOfDataSet):
    def __init__(self, n_task, n_grid):
        self.n_grid = n_grid
        super(SetOfMNISTPlusMNISTBPERM, self).__init__(n_task)

    def generate(self):
        self.list.append(ds.MNIST())
        self.list[0].flatten()
        for i in range(1, self.n_task):
            self.list.append(ds.RandMNISTBPERM(self.n_grid))


class SetOfCIFAR10PlusCIFAR10BPERM(SetOfDataSet):
    def __init__(self, n_task, n_grid):
        self.n_grid = n_grid
        super(SetOfCIFAR10PlusCIFAR10BPERM, self).__init__(n_task)

    def generate(self):
        self.list.append(ds.CIFAR10())
        self.list[0].flatten()
        for i in range(1, self.n_task):
            self.list.append(ds.RandCIFAR10BPERM(self.n_grid))


class SetOfMNISTPlusSwapMNISTBPERM(SetOfDataSet):
    def __init__(self, n_task, n_swap_epoch):
        n_pixel = 784
        self.perm = np.zeros(n_pixel * n_swap_epoch, dtype=int)
        for i in np.arange(n_swap_epoch):
            self.perm[n_pixel*i:n_pixel*(i+1)] = np.random.permutation(n_pixel)

        super(SetOfMNISTPlusSwapMNISTBPERM, self).__init__(n_task)

    def generate(self):
        self.list.append(ds.MNIST())
        self.list[0].flatten()
        for i in range(1, self.n_task):
            self.list.append(ds.SwapMnist(self.perm))


class SetOfRandMNISTPERM(SetOfDataSet):
    def __init__(self, n_task):
        super(SetOfRandMNISTPERM, self).__init__(n_task)

    def generate(self):
        for i in range(self.n_task):
            self.list.append(ds.RandMNISTPERM())


class SetOfRandRowMNISTPERM(SetOfDataSet):
    def __init__(self, n_task):
        super(SetOfRandRowMNISTPERM, self).__init__(n_task)

    def generate(self):
        for i in range(self.n_task):
            self.list.append(ds.RandRowMNISTPERM())


class SetOfRandColMNISTPERM(SetOfDataSet):
    def __init__(self, n_task):
        super(SetOfRandColMNISTPERM, self).__init__(n_task)

    def generate(self):
        for i in range(self.n_task):
            self.list.append(ds.RandColMNISTPERM())


class SetOfRandWholeMNISTPERM(SetOfDataSet):
    def __init__(self, n_task):
        super(SetOfRandWholeMNISTPERM, self).__init__(n_task)

    def generate(self):
        for i in range(self.n_task):
            self.list.append(ds.RandWholeMNISTPERM())


class SetOfRandMNISTBPERM(SetOfDataSet):
    def __init__(self, n_task, n_grid):
        self.n_grid = n_grid
        super(SetOfRandMNISTBPERM, self).__init__(n_task)

    def generate(self):
        for i in range(self.n_task):
            self.list.append(ds.RandMNISTBPERM(self.n_grid))


class SetOfRandMNISTROTA(SetOfDataSet):
    def __init__(self, n_task):
        super(SetOfRandMNISTROTA, self).__init__(n_task)

    def generate(self):
        for i in range(self.n_task):
            self.list.append(ds.RandMNISTROTA())


class SetOfGradualMNISTROTA(SetOfDataSet):
    def __init__(self, n_task, range):
        self.range = range
        super(SetOfGradualMNISTROTA, self).__init__(n_task)

    def generate(self):
        for i in range(self.n_task):
            angle = int((i + 1) * 360 * self.range / self.n_task)
            self.list.append(ds.MNISTROTA(angle))


class SetOfGradualMNISTSPLIT(SetOfDataSet):
    def __init__(self, n_task):
        self.period = int(10 / n_task)
        super(SetOfGradualMNISTSPLIT, self).__init__(n_task)

    def generate(self):
        for i in range(self.n_task):
            label_list = range(10)[self.period*i:self.period*(i+1)]
            print(label_list)
            self.list.append(ds.MNISTSPLIT(label_list))


class SetOfAlternativeMNIST(object):
    def __init__(self, n_task):
        self.list = []
        self.n_task = int(n_task/2)
        self.generate()

    def generate(self):
        for i in range(self.n_task):
            self.list.append(ds.RandMNISTPERM())
            self.list.append(ds.RandMNISTROTA())


class SetOfRandCIFAR10PERM(object):
    def __init__(self, n_task):
        self.list = []
        self.n_task = n_task
        self.generate()

    def generate(self):
        for i in range(self.n_task):
            self.list.append(ds.RandCIFAR10PERM())

    def concat(self):
        x_train_list = []
        y_train_list = []
        for i in range(self.n_task):
            x_train_list.append(self.list[i].x_train)
            y_train_list.append(self.list[i].y_train)

        multi_dataset = ds.RandCIFAR10PERM()
        multi_dataset.x_train = np.concatenate(x_train_list, axis=0)
        multi_dataset.y_train = np.concatenate(y_train_list, axis=0)

        return multi_dataset


class SetOfRandCIFAR10ROTA(SetOfDataSet):
    def __init__(self, n_task):
        super(SetOfRandCIFAR10ROTA, self).__init__(n_task)

    def generate(self):
        for i in range(self.n_task):
            self.list.append(ds.RandCIFAR10ROTA())


class SetOfRandCIFAR10BPERM(SetOfDataSet):
    def __init__(self, n_task, n_grid):
        self.n_grid = n_grid
        super(SetOfRandCIFAR10BPERM, self).__init__(n_task)

    def generate(self):
        for i in range(self.n_task):
            self.list.append(ds.RandCIFAR10BPERM(self.n_grid))

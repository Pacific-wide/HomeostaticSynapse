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

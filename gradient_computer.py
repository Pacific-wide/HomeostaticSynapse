import tensorflow as tf
import abc


class GradientComputer(object):
    def __init__(self, opt, loss):
        self.opt = opt
        self.loss = loss

    @abc.abstractmethod
    def compute(self):
        pass


class NormalGradientComputer(object):
    def __init__(self, opt, loss):
        super(GradientComputer, self).__init__(opt, loss)

    def compute(self):
        return self.opt.compute_gradient()

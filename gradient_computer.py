import tensorflow as tf
import abc


class GradientComputer(object):
    def __init__(self, opt, loss):
        self.opt = opt
        self.loss = loss

    @abc.abstractmethod
    def compute(self):
        pass


class NormalGradientComputer(GradientComputer):
    def __init__(self, opt, loss):
        super(NormalGradientComputer, self).__init__(opt, loss)

    def compute(self):
        return self.opt.compute_gradients(self.loss)


class ScopeGradientComputer(GradientComputer):
    def __init__(self, opt, loss, var_scope):
        super(ScopeGradientComputer, self).__init__(opt, loss)
        self.var_scope = var_scope

    def compute(self):
        return self.opt.compute_gradients(self.loss, var_list=self.var_scope)

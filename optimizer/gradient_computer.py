import tensorflow as tf
import abc


class GradientComputer(object):
    def __init__(self, opt, loss):
        self.opt = opt
        self.loss = loss

    @abc.abstractmethod
    def compute(self):
        pass


class ScopeGradientComputer(GradientComputer):
    def __init__(self, opt, loss, var_scope):
        super(ScopeGradientComputer, self).__init__(opt, loss)
        self.var_scope = var_scope

    def compute(self):
        return zip(self.opt.get_gradients(loss=self.loss, params=self.var_scope), self.var_scope)

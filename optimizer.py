import tensorflow as tf
import abc


class Optimizer(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    @abc.abstractmethod
    def build(self):
        pass


class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate):
        super(Optimizer, self).__init__(learning_rate)

    def build(self):
        return tf.train.GradientDescentOptimizer(self.learning_rate)


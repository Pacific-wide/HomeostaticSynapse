import tensorflow as tf
import abc


class Optimizer(object):
    @abc.abstractmethod
    def build(self, learning_rate):
        pass


class SGDOptimizer(Optimizer):
    def __init__(self):
        super(Optimizer, self).__init__()

    def build(self, learning_rate):
        return tf.train.GradientDescentOptimizer(learning_rate)


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
        super(SGDOptimizer, self).__init__(learning_rate)

    def build(self):
        return tf.keras.optimizers.SGD(self.learning_rate)


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate):
        super(Optimizer, self).__init__(learning_rate)

    def build(self):
        return tf.keras.optimizers.Adam(self.learning_rate)

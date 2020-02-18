import tensorflow as tf
import abc


class Loss(object):
    def __init__(self, prediction, label):
        self.prediction = prediction
        self.label = label

    @abc.abstractmethod
    def compute(self):
        pass


class SoftMaxCrossEntropyLoss(Loss):
    def __init__(self, prediction, label):
        super(SoftMaxCrossEntropyLoss, self).__init__(prediction, label)

    def compute(self):
        return tf.losses.softmax_cross_entropy(logits=self.prediction, onehot_labels=self.label)

import tensorflow as tf
import abc
import optimizer as op
import loss as ls
import gradient_computer as gc


class ModelFNCreator(object):
    def __init__(self, model, features, labels, mode, OptimizerSpec):
        self.logits = model(features)
        self.labels = labels
        self.mode = mode
        self.OptimizerSpec = OptimizerSpec

    @abc.abstractmethod
    def create(self):
        pass


class SingleModelFNCreator(ModelFNCreator):
    def __init__(self, model, features, labels, mode, params):
        super(ModelFNCreator, self).__init__(model, features, labels, mode)

    def create(self):

        one_hot_labels = tf.one_hot(self.labels, 10)

        opt = op.SGDOptimizer.build()
        loss = ls.SoftMaxCrossEntropyLoss(self.logits, one_hot_labels).compute()
        gradient_computer = gc.NormalGradientComputer(opt, loss)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            predictions = tf.argmax(self.logits, axis=1)
            accuracy = tf.metrics.accuracy(self.labels, predictions)
            return tf.estimator.EstimatorSpec(self.mode, loss=loss, eval_metric_ops={'accuracy': accuracy})

        train_op = opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(self.mode, loss=loss, train_op=train_op)








import tensorflow as tf
import abc
import loss as ls
import gradient_computer as gc


class ModelFNCreator(object):
    def __init__(self, model, features, labels, mode, OptimizerSpec):
        self.logits = model(features)
        self.predictions = tf.argmax(self.logits, axis=1)
        self.labels = labels
        self.mode = mode
        self.OptimizerSpec = OptimizerSpec
        self.opt = self.OptimizerSpec.optimizer

    @abc.abstractmethod
    def create(self):
        pass

    def pred_and_evaluate(self, mode, predictions):
        if mode == tf.estimator.ModeKeys.PREDICTs:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        if mode == tf.estimator.ModeKeys.EVAL:
            accuracy = tf.metrics.accuracy(self.labels, predictions)
            return tf.estimator.EstimatorSpec(self.mode, loss=loss, eval_metric_ops={'accuracy': accuracy})


class SingleModelFNCreator(ModelFNCreator):
    def __init__(self, model, features, labels, mode, OptimizerSpec):
        super(ModelFNCreator, self).__init__(model, features, labels, mode, OptimizerSpec)

    def create(self):
        one_hot_labels = tf.one_hot(self.labels, 10)
        loss = ls.SoftMaxCrossEntropyLoss(self.logits, one_hot_labels).compute()
        gradient_computer = gc.NormalGradientComputer(self.opt, loss)
        grads_and_vars = gradient_computer.compute()

        self.pred_create(self.mode, self.predictions)
        self.eval_create(self.mode, loss, self.predictions)

        train_op = self.opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(self.mode, loss=loss, train_op=train_op)








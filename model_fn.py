import tensorflow as tf
import abc
import loss as ls
import gradient_computer as gc
import net
import hook

class ModelFNCreator(object):
    def __init__(self, features, labels, mode, optimizer_spec):
        self.model = net.FCN()
        self.logits = self.model(features)
        self.predictions = tf.argmax(self.logits, axis=1)
        self.labels = labels
        self.mode = mode
        self.optimizer_spec = optimizer_spec

    @abc.abstractmethod
    def create(self):
        pass

    def evaluate(self, loss):
        accuracy = tf.metrics.accuracy(self.labels, self.predictions)
        metrics = {'accuracy': accuracy}
        tf.summary.scalar(name='accuracy', tensor=accuracy)
        return tf.estimator.EstimatorSpec(self.mode, loss=loss, eval_metric_ops=metrics)


class SingleModelFNCreator(ModelFNCreator):
    def __init__(self, features, labels, mode, optimizer_spec):
        super(SingleModelFNCreator, self).__init__(features, labels, mode, optimizer_spec)

    def create(self):
        one_hot_labels = tf.one_hot(self.labels, 10)
        loss = ls.SoftMaxCrossEntropyLoss(self.logits, one_hot_labels).compute()
        opt = self.optimizer_spec.optimizer
        gradient_computer = gc.NormalGradientComputer(opt, loss)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(loss)

        train_op = opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        gradient_hook = []
        for grad_and_var in grads_and_vars:
            gradient_hook.append(hook.SquareAccumulationGradientHook(grad_and_var))

        return tf.estimator.EstimatorSpec(self.mode, loss=loss, train_op=train_op, training_hooks=gradient_hook)


class EWCModelFNCreator(ModelFNCreator):
    def __init__(self, features, labels, mode, optimizer_spec):
        super(EWCModelFNCreator, self).__init__(features, labels, mode, optimizer_spec)

    def create(self):
        one_hot_labels = tf.one_hot(self.labels, 10)

        loss = ls.SoftMaxCrossEntropyLoss(self.logits, one_hot_labels).compute()

        loss = loss + self.add_ewc_loss

        opt = self.optimizer_spec.optimizer
        gradient_computer = gc.NormalGradientComputer(opt, loss)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(loss)

        train_op = opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(self.mode, loss=loss, train_op=train_op)

    def add_ewc_loss(self):

        ewc_loss = tf.losses.mean_squared_error(cur_var, pre_var, fisher)

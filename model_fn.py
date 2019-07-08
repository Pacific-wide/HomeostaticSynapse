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

        self.one_hot_labels = tf.one_hot(self.labels, 10)
        self.loss = ls.SoftMaxCrossEntropyLoss(self.logits, self.one_hot_labels).compute()
        self.opt = self.optimizer_spec.optimizer

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
        gradient_computer = gc.NormalGradientComputer(self.opt, self.loss)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(self.loss)

        train_op = self.opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        gradient_hook = []
        for grad_and_var in grads_and_vars:
            gradient_hook.append(hook.SquareAccumulationGradientHook(grad_and_var))

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=train_op, training_hooks=gradient_hook)


class EWCModelFNCreator(ModelFNCreator):
    def __init__(self, features, labels, mode, optimizer_spec):
        super(EWCModelFNCreator, self).__init__(features, labels, mode, optimizer_spec)

    def create(self):
        self.loss = self.loss + self.add_ewc_loss()
        gradient_computer = gc.NormalGradientComputer(self.opt, self.loss)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(self.loss)

        train_op = self.opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        gradient_hook = []
        for grad_and_var in grads_and_vars:
            gradient_hook.append(hook.SquareAccumulationGradientHook(grad_and_var))

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=train_op, training_hooks=gradient_hook)

    def add_ewc_loss(self, checkpoint="result"):
        ewc_loss = 0
        for w in self.model.weights:
            name = w.name[:-2]
            cur_var = w
            pre_var = tf.train.load_variable(checkpoint, name)
            fisher = tf.train.load_variable(checkpoint, 'fisher/' + name)
            ewc_loss = ewc_loss + tf.losses.mean_squared_error(cur_var, pre_var, fisher)

        return ewc_loss


class MetaModelFNCreator(ModelFNCreator):
    def __init__(self, features, labels, mode, optimizer_spec):
        super(MetaModelFNCreator, self).__init__(features, labels, mode, optimizer_spec)

    def create(self):
        self.loss = self.loss + self.add_ewc_loss()
        gradient_computer = gc.NormalGradientComputer(self.opt, self.loss)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(self.loss)

        train_op = self.opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        gradient_hook = []
        for grad_and_var in grads_and_vars:
            gradient_hook.append(hook.SquareAccumulationGradientHook(grad_and_var))

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=train_op, training_hooks=gradient_hook)

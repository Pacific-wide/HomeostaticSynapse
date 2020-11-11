import tensorflow as tf
import abc
from optimizer import gradient_computer as gc
from model import net
from model import hook
import numpy as np


class ModelFNCreator(object):
    def __init__(self, features, labels, mode, learning_spec):
        self.learning_spec = learning_spec
        self.optimizer_spec = self.learning_spec.optimizer_spec
        self.model = net.Main(self.optimizer_spec.d_in).build()
        self.features = features
        self.logits = self.model(features)
        self.predictions = tf.argmax(self.logits, axis=1)
        self.labels = labels
        self.mode = mode
        self.one_hot_labels = tf.one_hot(self.labels, 10)
        self.cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.loss = self.cce(self.one_hot_labels, self.logits)

        self.opt = self.optimizer_spec.optimizer.build()

        self.global_step = tf.compat.v1.train.get_global_step()

    @abc.abstractmethod
    def create(self):
        pass

    def load_tensors(self, checkpoint, prefix):
        pre_grads = []
        for i, w in enumerate(self.model.weights):
            name = w.name[5:-2]
            pre_grad = tf.train.load_variable(checkpoint, prefix + '/' + name)
            pre_grads.append(pre_grad)

        return pre_grads

    def evaluate(self, loss):
        accuracy = tf.keras.metrics.Accuracy()
        accuracy.update_state(self.labels, self.predictions)

        metrics = {'accuracy': accuracy}
        tf.summary.scalar(name='accuracy', data=accuracy.result())
        return tf.estimator.EstimatorSpec(self.mode, loss=loss, eval_metric_ops=metrics)

    def compute_curvature(self, grads_and_vars):
        gradient_hook = []
        for grad_and_var in grads_and_vars:
            n_total = self.learning_spec.n_batch * self.learning_spec.n_fed_step
            gradient_hook.append(hook.SquareAccumulationGradientHook(grad_and_var, self.learning_spec.n_batch, n_total))

        return gradient_hook

    def global_step_increase(self, grads_and_vars):
        global_step_increase_op = self.global_step.assign_add(1)
        with tf.control_dependencies([global_step_increase_op]):
            train_op = self.opt.apply_gradients(grads_and_vars)

        return train_op


class SingleModelFNCreator(ModelFNCreator):
    def __init__(self, features, labels, mode, learning_spec):
        super(SingleModelFNCreator, self).__init__(features, labels, mode, learning_spec)

    def create(self):
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(self.loss)

        train_op = self.global_step_increase(grads_and_vars)

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=train_op)


class BaseModelFNCreator(ModelFNCreator):
    def __init__(self, features, labels, mode, learning_spec):
        super(BaseModelFNCreator, self).__init__(features, labels, mode, learning_spec)

    def create(self):
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(self.loss)

        train_op = self.global_step_increase(grads_and_vars)

        gradient_hook = self.compute_curvature(grads_and_vars)

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=train_op, training_hooks=gradient_hook)


class CenterBaseModelFNCreator(BaseModelFNCreator):
    def __init__(self, features, labels, mode, learning_spec, i_task):
        super(CenterBaseModelFNCreator, self).__init__(features, labels, mode, learning_spec)
        self.i_task = i_task

    def compute_curvature(self, grads_and_vars):
        gradient_hook = []
        for i, grad_and_var in enumerate(grads_and_vars):
            gradient_hook.append(hook.CenterSquareAccumulationGradientHook(grad_and_var, self.learning_spec.n_batch, self.learning_spec.n_train))

        return gradient_hook


class IMMModelFNCreator(CenterBaseModelFNCreator):
    def __init__(self, features, labels, mode, learning_spec, i_task):
        super(IMMModelFNCreator, self).__init__(features, labels, mode, learning_spec, i_task)

    def evaluate(self, loss):
        v_center = self.load_tensors(self.learning_spec.model_dir, 'center')
        v_mean = self.average(v_center)

        self.model.set_weight(v_mean)
        self.logits = self.model(self.features)
        self.predictions = tf.argmax(self.logits, axis=1)

        accuracy = tf.metrics.accuracy(self.labels, self.predictions)
        metrics = {'accuracy': accuracy}
        tf.summary.scalar(name='accuracy', tensor=accuracy)

        return tf.estimator.EstimatorSpec(self.mode, loss=loss, eval_metric_ops=metrics)

    def average(self, v):
        average_v = []
        for item in v:
            average_v.append(item / (self.i_task+2))

        return average_v


class FullBaseModelFNCreator(BaseModelFNCreator):
    def __init__(self, features, labels, mode, learning_spec, i_task):
        super(FullBaseModelFNCreator, self).__init__(features, labels, mode, learning_spec)
        self.learning_spec = learning_spec
        self.i_task = i_task

    def compute_curvature(self, grads_and_vars):
        gradient_hook = []
        for i, grad_and_var in enumerate(grads_and_vars):
            gradient_hook.append(hook.SequentialSquareAccumulationGradientHook(grad_and_var, self.learning_spec.n_batch,
                                                                               self.learning_spec.n_train,
                                                                               self.learning_spec.n_task,
                                                                               self.i_task))

        return gradient_hook


class OEWCModelFNCreator(ModelFNCreator):
    def __init__(self, features, labels, mode, learning_spec):
        super(OEWCModelFNCreator, self).__init__(features, labels, mode, learning_spec)
        self.alpha = learning_spec.alpha

    def create(self):
        g_pre = self.load_tensors(self.learning_spec.model_dir, 'fisher')
        v_pre = self.load_tensors(self.learning_spec.model_dir, 'main')

        self.loss = self.loss + self.alpha * self.add_ewc_loss(self.model.weights, v_pre, g_pre)
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(self.loss)

        train_op = self.global_step_increase(grads_and_vars)

        gradient_hook = self.compute_curvature(grads_and_vars)

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=train_op, training_hooks=gradient_hook)

    def add_ewc_loss(self, v_cur, v_pre, g_pre):
        ewc_loss = 0
        for w, v, f in zip(v_cur, v_pre, g_pre):
            ewc_loss = ewc_loss + tf.math.reduce_sum(f * tf.math.square(w-v))

        return ewc_loss


class QuantizedEWCModelFNCreator(ModelFNCreator):
    def __init__(self, features, labels, mode, learning_spec):
        super(QuantizedEWCModelFNCreator, self).__init__(features, labels, mode, learning_spec)
        self.alpha = learning_spec.alpha

    def create(self):
        g_pre = self.load_tensors(self.learning_spec.model_dir, 'fisher')
        v_pre = self.load_tensors(self.learning_spec.model_dir, 'main')

        self.loss = self.loss + self.alpha * self.add_ewc_loss(self.model.weights, v_pre, g_pre)
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(self.loss)

        train_op = self.global_step_increase(grads_and_vars)

        gradient_hook = self.compute_curvature(grads_and_vars)

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=train_op, training_hooks=gradient_hook)

    def add_ewc_loss(self, v_cur, v_pre, g_pre):
        ewc_loss = 0
        for w, v, f in zip(v_cur, v_pre, g_pre):
            quantized_f = np.round(f, 1)
            ewc_loss = ewc_loss + tf.math.reduce_sum(quantized_f*tf.math.square(w-v))

        return ewc_loss


class CenterEWCModelFNCreator(CenterBaseModelFNCreator):
    def __init__(self, features, labels, mode, learning_spec, i_task):
        super(CenterEWCModelFNCreator, self).__init__(features, labels, mode, learning_spec, i_task)
        self.alpha = learning_spec.alpha

    def create(self):
        g_pre = self.load_tensors(self.learning_spec.model_dir, 'fisher')
        v_pre = self.load_tensors(self.learning_spec.model_dir, 'center')

        self.loss = self.loss + self.alpha * self.add_ewc_loss(self.model.weights, v_pre, g_pre)
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(self.loss)

        train_op = self.global_step_increase(grads_and_vars)

        gradient_hook = self.compute_curvature(grads_and_vars)

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=train_op, training_hooks=gradient_hook)

    def add_ewc_loss(self, v_cur, v_pre, g_pre):
        ewc_loss = 0
        for w, v, f in zip(v_cur, v_pre, g_pre):
            v = v / (self.i_task + 1)
            ewc_loss = ewc_loss + tf.losses.mean_squared_error(w, v, f)

        return ewc_loss


class EWCModelFNCreator(FullBaseModelFNCreator):
    def __init__(self, features, labels, mode, learning_spec, i_task):
        super(EWCModelFNCreator, self).__init__(features, labels, mode, learning_spec, i_task)
        self.alpha = learning_spec.alpha

    def create(self):
        self.loss = self.loss + self.alpha * self.add_ewc_loss(self.model.weights)
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(self.loss)

        train_op = self.global_step_increase(grads_and_vars)

        gradient_hook = self.compute_curvature(grads_and_vars)

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=train_op, training_hooks=gradient_hook)

    def add_ewc_loss(self, v_cur):
        ewc_loss = 0
        for i in range(self.i_task):
            g_pre = self.load_tensors(self.learning_spec.model_dir, 'fisher'+str(i))
            v_pre = self.load_tensors(self.learning_spec.model_dir, 'theta'+str(i))
            for w, v, f in zip(v_cur, v_pre, g_pre):
                ewc_loss = ewc_loss + tf.losses.mean_squared_error(w, v, f)

        return ewc_loss


class QEWCModelFNCreator(EWCModelFNCreator):
    def __init__(self, features, labels, mode, learning_spec, i_task):
        super(EWCModelFNCreator, self).__init__(features, labels, mode, learning_spec, i_task)

    def create(self):
        self.loss = self.loss + self.alpha * self.add_ewc_loss(self.model.weights)
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(self.loss)

        train_op = self.global_step_increase(grads_and_vars)

        gradient_hook = self.compute_curvature(grads_and_vars)

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=train_op, training_hooks=gradient_hook)

    def add_ewc_loss(self, v_cur):
        ewc_loss = 0
        for i in range(self.i_task):
            g_pre = self.load_tensors(self.learning_spec.model_dir, 'fisher'+str(i))
            v_pre = self.load_tensors(self.learning_spec.model_dir, 'theta'+str(i))
            for w, v, f in zip(v_cur, v_pre, g_pre):
                ewc_loss = ewc_loss + tf.losses.mean_squared_error(w, v, f)

        return ewc_loss
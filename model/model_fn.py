import tensorflow as tf
import abc
from optimizer import loss as ls
from optimizer import gradient_computer as gc
from model import net
from model import hook
import numpy as np


class ModelFNCreator(object):
    def __init__(self, features, labels, mode, learning_spec):
        self.learning_spec = learning_spec
        self.optimizer_spec = self.learning_spec.optimizer_spec
        self.model = net.Main(self.optimizer_spec.d_in).build()
        print(features)
        self.features = features
        self.logits = self.model(features)
        self.predictions = tf.argmax(self.logits, axis=1)
        self.labels = labels
        self.mode = mode

        self.one_hot_labels = tf.one_hot(self.labels, 10)
        self.loss = ls.SoftMaxCrossEntropyLoss(self.logits, self.one_hot_labels).compute()
        self.opt = self.optimizer_spec.optimizer

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
        accuracy = tf.metrics.accuracy(self.labels, self.predictions)
        metrics = {'accuracy': accuracy}
        tf.summary.scalar(name='accuracy', tensor=accuracy)
        return tf.estimator.EstimatorSpec(self.mode, loss=loss, eval_metric_ops=metrics)

    def compute_curvature(self, grads_and_vars):
        gradient_hook = []
        for i, grad_and_var in enumerate(grads_and_vars):
            gradient_hook.append(hook.SquareAccumulationGradientHook(grad_and_var, self.learning_spec.n_batch))

        return gradient_hook


class SingleModelFNCreator(ModelFNCreator):
    def __init__(self, features, labels, mode, learning_spec):
        super(SingleModelFNCreator, self).__init__(features, labels, mode, learning_spec)

    def create(self):
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(self.loss)

        train_op = self.opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=train_op)


class BaseModelFNCreator(ModelFNCreator):
    def __init__(self, features, labels, mode, learning_spec):
        super(BaseModelFNCreator, self).__init__(features, labels, mode, learning_spec)

    def create(self):
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(self.loss)

        train_op = self.opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        gradient_hook = self.compute_curvature(grads_and_vars)

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=train_op, training_hooks=gradient_hook)


class CenterBaseModelFNCreator(BaseModelFNCreator):
    def __init__(self, features, labels, mode, learning_spec, i_task):
        super(CenterBaseModelFNCreator, self).__init__(features, labels, mode, learning_spec)
        self.i_task = i_task

    def compute_curvature(self, grads_and_vars):
        gradient_hook = []
        for i, grad_and_var in enumerate(grads_and_vars):
            gradient_hook.append(hook.CenterSquareAccumulationGradientHook(grad_and_var, self.learning_spec.n_batch))

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
                                                                               self.learning_spec.n_task,
                                                                               self.i_task))

        return gradient_hook


class EWCModelFNCreator(ModelFNCreator):
    def __init__(self, features, labels, mode, learning_spec):
        super(EWCModelFNCreator, self).__init__(features, labels, mode, learning_spec)
        self.alpha = learning_spec.alpha

    def create(self):
        g_pre = self.load_tensors(self.learning_spec.model_dir, 'fisher')
        v_pre = self.load_tensors(self.learning_spec.model_dir, 'main')

        print(g_pre)

        self.loss = self.loss + self.alpha * self.add_ewc_loss(self.model.weights, v_pre, g_pre)
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(self.loss)

        train_op = self.opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        gradient_hook = self.compute_curvature(grads_and_vars)

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=train_op, training_hooks=gradient_hook)

    def add_ewc_loss(self, v_cur, v_pre, g_pre):
        ewc_loss = 0
        for w, v, f in zip(v_cur, v_pre, g_pre):
            ewc_loss = ewc_loss + tf.losses.mean_squared_error(w, v, f)

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

        train_op = self.opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        gradient_hook = self.compute_curvature(grads_and_vars)

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=train_op, training_hooks=gradient_hook)

    def add_ewc_loss(self, v_cur, v_pre, g_pre):
        ewc_loss = 0
        for w, v, f in zip(v_cur, v_pre, g_pre):
            v = v / (self.i_task + 1)
            ewc_loss = ewc_loss + tf.losses.mean_squared_error(w, v, f)

        return ewc_loss


class FullEWCModelFNCreator(FullBaseModelFNCreator):
    def __init__(self, features, labels, mode, learning_spec, i_task):
        super(FullEWCModelFNCreator, self).__init__(features, labels, mode, learning_spec, i_task)
        self.alpha = learning_spec.alpha

    def create(self):
        self.loss = self.loss + self.alpha * self.add_ewc_loss(self.model.weights)
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(self.loss)

        train_op = self.opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

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


class MetaModelFNCreator(ModelFNCreator):
    def __init__(self, features, labels, mode, learning_spec):
        super(MetaModelFNCreator, self).__init__(features, labels, mode, learning_spec)

    def create(self):
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(self.loss)

        train_op = self.opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        gradient_hook = self.compute_curvature(grads_and_vars)

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=train_op, training_hooks=gradient_hook)

    def add_meta_loss(self, g_pre, v_cur, v_pre):
        meta_loss = 0
        for (w, f, v) in zip(v_cur, g_pre, v_pre):

            meta_loss = meta_loss + tf.square(tf.losses.mean_squared_error(w, v, f))

        return meta_loss

    @staticmethod
    def layer_to_flat(grads):
        grad_list = []
        for grad in grads:
            flat_grad = tf.reshape(grad, [-1, 1])
            grad_list.append(flat_grad)

        return tf.concat(grad_list, axis=0)

    @staticmethod
    def flat_to_layer(grads, cur_vars):
        grads_list = []
        begin = 0
        for var in cur_vars:
            pre_shape = tf.shape(var)
            flat_var = tf.reshape(var, [-1, 1])
            size = tf.shape(flat_var)[0]
            grad_vec = tf.reshape(grads, [-1])
            flat_grad = tf.slice(grad_vec, [begin], [size])
            grad = tf.reshape(flat_grad, pre_shape)
            grads_list.append(grad)
            begin = begin + size

        return grads_list


class MetaAlphaModelFNCreator(MetaModelFNCreator):
    def __init__(self, features, labels, mode, learning_spec):
        super(MetaAlphaModelFNCreator, self).__init__(features, labels, mode, learning_spec)
        self.meta_model = net.MetaAlpha().build()

    def combine_meta_features(self, g_cur, g_pre, v_pre, v_cur):
        flat_g_pre = self.layer_to_flat(g_pre)
        flat_g_cur = self.layer_to_flat(g_cur)
        flat_v_pre = self.layer_to_flat(v_pre)
        flat_v_cur = self.layer_to_flat(v_cur)

        combine_list = (tf.reduce_sum(tf.matmul(flat_g_pre, flat_g_cur, transpose_a=True)),
                        tf.norm(flat_v_cur-flat_v_pre))

        return tf.reshape(tf.stack(combine_list), shape=[1, -1])


class MetaAlphaTestModelFNCreator(MetaAlphaModelFNCreator):
    def __init__(self, features, labels, mode, learning_spec):
        super(MetaAlphaTestModelFNCreator, self).__init__(features, labels, mode, learning_spec)

    def create(self):
        # current gradient
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        g_cur, v_cur = zip(*grads_and_vars)
        g_pre = self.load_tensors(self.learning_spec.model_dir, 'fisher')
        v_pre = self.load_tensors(self.learning_spec.model_dir, 'main')

        meta_batch = self.combine_meta_features(g_cur, g_pre, v_pre, v_cur)
        meta_output = self.meta_model(meta_batch)

        tf.summary.scalar(name='losses/meta_output', tensor=tf.reshape(meta_output, shape=[]))

        self.total_loss = self.loss + meta_output * self.add_meta_loss(g_pre, self.model.weights, v_pre)
        tf.summary.scalar(name='losses/total_loss', tensor=tf.reshape(self.total_loss, shape=[]))

        total_gradient_computer = gc.ScopeGradientComputer(self.opt, self.total_loss, self.model.weights)
        total_grads_and_vars = total_gradient_computer.compute()

        train_op = self.opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        gradient_hook = self.compute_curvature(total_grads_and_vars)

        return tf.estimator.EstimatorSpec(self.mode, loss=self.total_loss, train_op=train_op, training_hooks=gradient_hook)


class MetaAlphaTrainModelFNCreator(MetaAlphaModelFNCreator):
    def __init__(self, features, labels, mode, learning_spec, meta_learning_spec):
        self.meta_learning_spec = meta_learning_spec
        self.meta_optimizer_spec = self.meta_learning_spec.optimizer_spec
        self.meta_opt = self.meta_optimizer_spec.optimizer

        _, self.cur_features, self.joint_features = features
        _, self.cur_labels, self.joint_labels = labels

        super(MetaAlphaTrainModelFNCreator, self).__init__(self.cur_features, self.cur_labels, mode, learning_spec)

        # joint gradient
        self.joint_logits = self.model(self.joint_features)
        self.joint_one_hot_labels = tf.one_hot(self.joint_labels, 10)
        self.joint_loss = tf.losses.softmax_cross_entropy(self.joint_one_hot_labels, self.joint_logits)

    def create(self):
        # current gradient
        tf.summary.scalar(name='losses/cur_loss', tensor=self.loss)
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        g_cur, v_cur = zip(*grads_and_vars)

        joint_gradient_computer = gc.ScopeGradientComputer(self.opt, self.joint_loss, self.model.weights)
        joint_grads_and_vars = joint_gradient_computer.compute()

        g_joint, _ = zip(*joint_grads_and_vars)
        g_pre = self.load_tensors(self.meta_learning_spec.model_dir, 'fisher')
        v_pre = self.load_tensors(self.meta_learning_spec.model_dir, 'main')

        train_op = self.opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        meta_batch = self.combine_meta_features(g_cur, g_pre, v_cur, v_pre)
        meta_label = self.make_meta_labels(g_cur, g_joint, v_cur, v_pre, g_pre)
        tf.summary.scalar(name='losses/meta_label', tensor=tf.reshape(meta_label, []))
        meta_output = self.meta_model(meta_batch)
        tf.summary.scalar(name='losses/meta_output', tensor=tf.reshape(meta_output, []))

        meta_loss = tf.losses.absolute_difference(meta_output, meta_label)
        tf.summary.scalar(name='losses/meta_loss', tensor=meta_loss)

        meta_gradient_computer = gc.ScopeGradientComputer(self.meta_opt, meta_loss, self.meta_model.weights)
        meta_grads_and_vars = meta_gradient_computer.compute()

        meta_train_op = self.meta_opt.apply_gradients(meta_grads_and_vars, global_step=tf.train.get_global_step())

        gradient_hook = self.compute_curvature(grads_and_vars)

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=tf.group([train_op, meta_train_op]),
                                          training_hooks=gradient_hook)

    def make_meta_labels(self, g_cur, g_joint, v_cur, v_pre, g_pre):
        flat_g_pre = self.layer_to_flat(g_pre)
        flat_g_cur = self.layer_to_flat(g_cur)
        flat_v_pre = self.layer_to_flat(v_pre)
        flat_v_cur = self.layer_to_flat(v_cur)

        flat_g_joint = self.layer_to_flat(g_joint)

        X = flat_g_joint-flat_g_cur
        Y = tf.multiply(flat_g_pre, (flat_v_cur-flat_v_pre))
        X2 = tf.multiply(X, X)
        XY = tf.multiply(X, Y)

        epsilon = 1e-3

        alpha = tf.reduce_sum(X2)/(tf.reduce_sum(XY) + epsilon)
        tf.summary.scalar(name='parameter/alpha', tensor=alpha)

        return tf.reshape(alpha, shape=[1, -1])

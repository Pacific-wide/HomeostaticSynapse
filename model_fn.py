import tensorflow as tf
import abc
import loss as ls
import gradient_computer as gc
import net
import hook
import numpy as np


class ModelFNCreator(object):
    def __init__(self, features, labels, mode, optimizer_spec):
        self.model = net.Main(optimizer_spec.d_in).build()
        self.features = features
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
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(self.loss)

        train_op = self.opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=train_op)


class BaseModelFNCreator(ModelFNCreator):
    def __init__(self, features, labels, mode, optimizer_spec):
        super(BaseModelFNCreator, self).__init__(features, labels, mode, optimizer_spec)

    def create(self):
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(self.loss)

        train_op = self.opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        gradient_hook = []
        for i, grad_and_var in enumerate(grads_and_vars):
            gradient_hook.append(hook.SquareAccumulationGradientHook(grad_and_var))

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=train_op, training_hooks=gradient_hook)


class EWCModelFNCreator(ModelFNCreator):
    def __init__(self, features, labels, mode, optimizer_spec, learning_spec):
        super(EWCModelFNCreator, self).__init__(features, labels, mode, optimizer_spec)
        self.alpha = 0.01
        self.learning_spec = learning_spec

    def create(self):
        self.loss = self.loss + self.alpha * self.add_ewc_loss(self.learning_spec.model_dir)
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(self.loss)

        train_op = self.opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        gradient_hook = []
        for grad_and_var in grads_and_vars:
            gradient_hook.append(hook.SquareAccumulationGradientHook(grad_and_var))

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=train_op, training_hooks=gradient_hook)

    def add_ewc_loss(self, checkpoint):
        ewc_loss = 0
        for i, w in enumerate(self.model.weights):
            name = w.name[:-2]
            layer_name = w.name[5:-2]
            cur_var = w
            pre_var = tf.train.load_variable(checkpoint, name)
            fisher = tf.train.load_variable(checkpoint, 'fisher/' + layer_name)
            ewc_loss = ewc_loss + tf.losses.mean_squared_error(cur_var, pre_var, fisher)

        return ewc_loss


class MetaModelFNCreator(ModelFNCreator):
    def __init__(self, features, labels, mode, optimizer_spec):
        super(MetaModelFNCreator, self).__init__(features, labels, mode, optimizer_spec)
        self.meta_model = net.Meta().build()

    def create(self):
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        if self.mode == tf.estimator.ModeKeys.EVAL:
            return self.evaluate(self.loss)

        train_op = self.opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        gradient_hook = []
        for grad_and_var in grads_and_vars:
            gradient_hook.append(hook.SquareAccumulationGradientHook(grad_and_var))

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=train_op, training_hooks=gradient_hook)

    def load_tensors(self, checkpoint, prefix):
        pre_grads = []
        for i, w in enumerate(self.model.weights):
            name = w.name[5:-2]
            pre_grad = tf.train.load_variable(checkpoint, prefix + '/' + name)
            pre_grads.append(pre_grad)

        return pre_grads

    @staticmethod
    def layer_to_flat(grads):
        grad_list = []
        for grad in grads:
            flat_grad = tf.reshape(grad, [-1, 1])
            grad_list.append(flat_grad)

        return tf.concat(grad_list, axis=0)

    @staticmethod
    def flat_to_layer(grads, cur_vars):
        grads_and_vars = []
        for var in cur_vars:
            pre_shape = tf.shape(var)
            flat_var = tf.reshape(var, [-1, 1])
            size = tf.shape(flat_var)[0]
            begin = 0
            grad_vec = tf.reshape(grads, [-1])
            flat_grad = tf.slice(grad_vec, [begin], [size])
            grad = tf.reshape(flat_grad, pre_shape)
            grads_and_vars.append((grad, var))
            begin = begin + size

        return grads_and_vars


class MetaTestModelFNCreator(MetaModelFNCreator):
    def __init__(self, features, labels, mode, optimizer_spec, learning_spec):
        super(MetaTestModelFNCreator, self).__init__(features, labels, mode, optimizer_spec)

        self.learning_spec = learning_spec
        self.alpha = 0.001

    def create(self):
        # current gradient
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        g_cur, v_cur = zip(*grads_and_vars)
        g_pre = self.load_tensors(self.learning_spec.model_dir, 'fisher')

        meta_batch = self.combine_meta_features(g_cur, g_pre, v_cur)
        meta_output = self.meta_model(meta_batch)

        self.total_loss = self.loss + self.alpha * self.add_meta_loss(meta_output)
        total_gradient_computer = gc.ScopeGradientComputer(self.opt, self.total_loss, self.model.weights)
        total_grads_and_vars = total_gradient_computer.compute()

        train_op = self.opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        gradient_hook = []
        for total_grad_and_var in total_grads_and_vars:
            gradient_hook.append(hook.SquareAccumulationGradientHook(total_grad_and_var))

        return tf.estimator.EstimatorSpec(self.mode, loss=self.total_loss, train_op=train_op, training_hooks=gradient_hook)

    def add_meta_loss(self, meta_output):
        meta_loss = 0
        layer_meta_output = self.flat_to_layer(meta_output, self.model.weights)

        for (w, m) in zip(self.model.weights, layer_meta_output):
            name = w.name[:-2]
            cur_var = w
            pre_var = tf.train.load_variable("meta", name)
            fisher = m[0]
            meta_loss = meta_loss + tf.losses.mean_squared_error(cur_var, pre_var, fisher)

        return meta_loss

    def combine_meta_features(self, cur, pre, weight):
        combine_list = (self.layer_to_flat(cur), self.layer_to_flat(pre), self.layer_to_flat(weight))

        return tf.concat(combine_list, axis=1)


class MetaTrainModelFNCreator(MetaModelFNCreator):
    def __init__(self, features, labels, mode, optimizer_spec, meta_learning_spec):
        _, self.cur_features, self.joint_features = features
        _, self.cur_labels, self.joint_labels = labels
        self.alpha = 0.1

        super(MetaTrainModelFNCreator, self).__init__(self.cur_features, self.cur_labels, mode, optimizer_spec)

        # joint gradient
        self.joint_logits = self.model(self.joint_features)
        self.joint_one_hot_labels = tf.one_hot(self.joint_labels, 10)
        self.joint_loss = tf.losses.softmax_cross_entropy(self.joint_one_hot_labels, self.joint_logits)
        tf.summary.scalar(name='losses/joint_loss', tensor=self.joint_loss)

        self.meta_learning_spec = meta_learning_spec
        self.meta_optimizer_spec = self.meta_learning_spec.optimizer_spec
        self.meta_opt = self.meta_optimizer_spec.optimizer

    def create(self):
        # current gradient
        tf.summary.scalar(name='losses/cur_loss', tensor=self.loss)
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        joint_gradient_computer = gc.ScopeGradientComputer(self.opt, self.joint_loss, self.model.weights)
        joint_grads_and_vars = joint_gradient_computer.compute()

        g_cur, v_cur = zip(*grads_and_vars)
        g_joint, _ = zip(*joint_grads_and_vars)

        g_pre = self.load_tensors(self.meta_learning_spec.model_dir, 'fisher')
        v_pre = self.load_tensors(self.meta_learning_spec.model_dir, 'main')

        train_op = self.opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        meta_batch = self.combine_meta_features(g_cur, g_pre, v_cur)
        meta_label = self.make_meta_labels(g_cur, g_joint, v_cur, v_pre, self.alpha)

        meta_output = self.meta_model(meta_batch)
        meta_loss = tf.losses.mean_squared_error(meta_output, meta_label)
        tf.summary.scalar(name='losses/meta_loss', tensor=meta_loss)

        meta_gradient_computer = gc.ScopeGradientComputer(self.meta_opt, meta_loss, self.meta_model.weights)
        meta_grads_and_vars = meta_gradient_computer.compute()

        meta_train_op = self.meta_opt.apply_gradients(meta_grads_and_vars, global_step=tf.train.get_global_step())

        gradient_hook = []
        for grad_and_var in grads_and_vars:
            gradient_hook.append(hook.SquareAccumulationGradientHook(grad_and_var))

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=tf.group([train_op, meta_train_op]),
                                          training_hooks=gradient_hook)

    def combine_meta_features(self, g_cur, g_pre, t_cur):
        combine_list = (self.layer_to_flat(g_cur), self.layer_to_flat(g_pre), self.layer_to_flat(t_cur))

        return tf.concat(combine_list, axis=1)

    def make_meta_labels(self, g_cur, g_joint, t_cur, t_pre, alpha):
        flat_g_cur = self.layer_to_flat(g_cur)
        flat_g_joint = self.layer_to_flat(g_joint)
        flat_t_cur = self.layer_to_flat(t_cur)
        flat_t_pre = self.layer_to_flat(t_pre)
        epsilon = 5e-1

        importance = 0.5 * (flat_g_joint - flat_g_cur) / (alpha * abs(flat_t_cur - flat_t_pre + epsilon))
        return importance


class LayerwiseMetaModelFNCreator(MetaTrainModelFNCreator):
    def __init__(self, features, labels, mode, optimizer_spec, meta_optimizer_spec):
        super(MetaTrainModelFNCreator, self).__init__(self.cur_features, self.cur_labels, mode, optimizer_spec)

    def create(self):
        self.meta_model = net.MultiFCN("meta", n_layer=4, n_input=3, n_output=1, n_unit=30, n_layer_main=2)

        # current gradient
        tf.summary.scalar(name='losses/cur_loss', tensor=self.loss)
        gradient_computer = gc.ScopeGradientComputer(self.opt, self.loss, self.model.weights)
        grads_and_vars = gradient_computer.compute()

        joint_gradient_computer = gc.ScopeGradientComputer(self.opt, self.joint_loss, self.model.weights)
        joint_grads_and_vars = joint_gradient_computer.compute()

        grads, vars = zip(*grads_and_vars)
        joint_grads, joint_vars = zip(*joint_grads_and_vars)
        pre_grads = self.load_pre_grads(self.optimizer_spec.pre_model_dir)

        train_op = self.opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        meta_batch = self.combine_meta_features(grads, pre_grads, self.model.weights)
        meta_label = self.layer_to_flat(joint_grads)

        meta_output = self.meta_model(meta_batch)
        meta_loss = tf.losses.mean_squared_error(meta_output, meta_label)
        tf.summary.scalar(name='losses/meta_loss', tensor=meta_loss)

        meta_gradient_computer = gc.ScopeGradientComputer(self.meta_opt, meta_loss, self.meta_model.weights)
        meta_grads_and_vars = meta_gradient_computer.compute()

        meta_train_op = self.meta_opt.apply_gradients(meta_grads_and_vars, global_step=tf.train.get_global_step())

        gradient_hook = []
        for grad_and_var in grads_and_vars:
            gradient_hook.append(hook.SquareAccumulationGradientHook(grad_and_var))

        return tf.estimator.EstimatorSpec(self.mode, loss=self.loss, train_op=tf.group([train_op, meta_train_op]),
                                          training_hooks=gradient_hook)

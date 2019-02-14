import os
import numpy as np
import tensorflow as tf


def base(features, labels, mode, params):
    model = FullyConnectedNetwork()
    logits = model(features)
    predictions = tf.argmax(logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        softmax_layer = tf.keras.layers.Softmax()
        probabilities = softmax_layer(logits)
        return tf.estimator.EstimatorSpec(mode, predictions={'predictions': predictions, 'probabilities': probabilities})

    one_hot_labels = tf.one_hot(labels, 10)
    loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels, predictions)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'accuracy': accuracy})

    opt = tf.train.GradientDescentOptimizer(learning_rate=params['lr'])
    grads_and_vars = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

    meta_model = MetaNetwork()

    fisher_hook = []
    for grad_and_var in grads_and_vars:
        fisher_hook.append(GradientAccumulationHook(grad_and_var))

    return tf.estimator.EstimatorSpec(mode, loss=loss, training_hooks=fisher_hook, train_op=train_op)


def ewc(features, labels, mode, params):
    model = FullyConnectedNetwork()
    logits = model(features)
    predictions = tf.argmax(logits, axis=1)
    one_hot_labels = tf.one_hot(labels, 10)

    if mode == tf.estimator.ModeKeys.PREDICT:
        softmax_layer = tf.keras.layers.Softmax()
        probabilities = softmax_layer(logits)
        return tf.estimator.EstimatorSpec(mode, predictions={'predictions': predictions, 'probabilities': probabilities})

    loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)

    checkpoint = params['eval_model_dir']
    if os.path.isdir(checkpoint):
        for weight in model.weights:
            name = weight.name[:-2]
            cur_var = weight
            pre_var = tf.train.load_variable(checkpoint, name)
            fisher = tf.train.load_variable(checkpoint, name+'/fisher')

            ewc_loss = tf.losses.mean_squared_error(cur_var, pre_var, fisher)
            loss = loss + ewc_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels, predictions)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'accuracy': accuracy})

    opt = tf.train.GradientDescentOptimizer(learning_rate=params['lr'])

    grads_and_vars = opt.compute_gradients(loss)
    fisher_hook = []
    for grad_and_var in grads_and_vars:
        fisher_hook.append(GradientAccumulationHook(grad_and_var))

    train_op = opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, training_hooks=fisher_hook, train_op=train_op)


def meta_train(features, labels, mode, params):
    model = FullyConnectedNetwork()

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_logits = model(features)
        predictions = tf.argmax(eval_logits, axis=1)
        one_hot_labels = tf.one_hot(labels, 10)
        eval_loss = tf.losses.softmax_cross_entropy(one_hot_labels, eval_logits)
        accuracy = tf.metrics.accuracy(labels, predictions)
        return tf.estimator.EstimatorSpec(mode, loss=eval_loss, eval_metric_ops={'accuracy': accuracy})

    joint_features, single_features0, single_features1 = features
    joint_labels, single_labels0, single_labels1 = labels

    # single 0
    logits0 = model(single_features0)
    one_hot_labels0 = tf.one_hot(single_labels0, 10)
    loss0 = tf.losses.softmax_cross_entropy(one_hot_labels0, logits0)
    tf.summary.scalar(name='losses/main_loss0', tensor=loss0)

    # single 1
    logits1 = model(single_features1)
    one_hot_labels1 = tf.one_hot(single_labels1, 10)
    loss1 = tf.losses.softmax_cross_entropy(one_hot_labels1, logits1)
    tf.summary.scalar(name='losses/main_loss1', tensor=loss1)

    # joint
    joint_logits = model(joint_features)
    joint_one_hot_labels = tf.one_hot(joint_labels, 10)
    joint_loss = tf.losses.softmax_cross_entropy(joint_one_hot_labels, joint_logits)

    opt = tf.train.GradientDescentOptimizer(learning_rate=params['lr'])

    grads_and_vars0 = opt.compute_gradients(loss0)
    grads_and_vars1 = opt.compute_gradients(loss1)
    joint_grads_and_vars = opt.compute_gradients(joint_loss)

    grad0, var0 = zip(*grads_and_vars0)
    grad1, var1 = zip(*grads_and_vars1)
    joint_grad, joint_var = zip(*joint_grads_and_vars)

    train_op = opt.apply_gradients(grads_and_vars0, global_step=tf.train.get_global_step())

    meta_model = MetaNetwork()
    meta_batch = combine_meta_batch(grad0, grad1, model.weights)
    meta_label = layer_to_flat(joint_grad)

    meta_output = meta_model(meta_batch)
    meta_loss = tf.losses.mean_squared_error(meta_output, meta_label)
    tf.summary.scalar(name='losses/meta_loss', tensor=meta_loss)

    meta_grads_and_vars = opt.compute_gradients(meta_loss, var_list=meta_model.weights)
    meta_grads, meta_var = zip(*meta_grads_and_vars)
    normalized_meta_grads = []
    for meta_grad in meta_grads:
        normalized_meta_grads.append(tf.math.l2_normalize(meta_grad))
    meta_grads_and_vars = zip(normalized_meta_grads, meta_var)

    meta_opt = tf.train.GradientDescentOptimizer(learning_rate=params['meta_lr'])

    meta_train_op = meta_opt.apply_gradients(meta_grads_and_vars, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=meta_loss, train_op=tf.group([train_op, meta_train_op]))


def meta_eval(features, labels, mode, params):
    model = FullyConnectedNetwork()
    logits = model(features)
    predictions = tf.argmax(logits, axis=1)

    one_hot_labels = tf.one_hot(labels, 10)
    loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)

    checkpoint = params["eval_model_dir"]
    fishers = []
    for weight in model.weights:
        name = weight.name[:-2]
        fisher = tf.train.load_variable(checkpoint, name+'/fisher')
        fishers.append(fisher)

    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels, predictions)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'accuracy': accuracy})

    opt = tf.train.GradientDescentOptimizer(learning_rate=params['lr'])
    grads_and_vars = opt.compute_gradients(loss)

    fisher_hook = []
    for grad_and_var in grads_and_vars:
        fisher_hook.append(GradientAccumulationHook(grad_and_var))

    grad, var = zip(*grads_and_vars)
    #grad, _ = tf.clip_by_global_norm(grad, 1.0)

    meta_model = MetaNetwork()
    pre_grad_square = fishers

    meta_batch = combine_meta_batch(grad, pre_grad_square, model.weights)
    meta_output = meta_model(meta_batch)

    final_grads_and_vars = flat_to_layer(meta_output, var)

    train_op = opt.apply_gradients(final_grads_and_vars, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, training_hooks=fisher_hook, train_op=train_op)


def combine_meta_batch(cur, pre, weight):
    combine_list = [layer_to_flat(cur),
                    layer_to_flat(pre),
                    layer_to_flat(weight)]

    return tf.concat(combine_list, axis=1)


def layer_to_flat(grads):
    grad_list = []
    for grad in grads:
        flat_grad = tf.reshape(grad, [-1, 1])
        grad_list.append(flat_grad)

    return tf.concat(grad_list, axis=0)


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


class FullyConnectedNetwork(tf.keras.models.Model):
    def __init__(self):
        super(FullyConnectedNetwork, self).__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.InputLayer((784,)),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(10)])

    def call(self, inputs):
        return self.net(inputs)


class MetaNetwork(tf.keras.models.Model):
    def __init__(self):
        super(MetaNetwork, self).__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.InputLayer((3,)),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)])

    def call(self, inputs):
        return self.net(inputs) + inputs[:, 0:1]


class GradientAccumulationHook(tf.train.SessionRunHook):
    def __init__(self, grad_and_var):
        self.gradients = grad_and_var[0]
        self.variable = grad_and_var[1]
        self.name = self.variable.name[:-2]

    def begin(self):
        self.global_step = tf.train.get_global_step()
        # self.sum_operation = self.sum_gradients.assign_add(tf.math.square(self.gradients))
        self.mean_gradients_metric = tf.metrics.mean_tensor(self.gradients)
        self.mean_gradients = tf.Variable(tf.zeros_like(self.gradients), name=(self.name + '/fisher'))
        self.assign_mean_gradients = self.mean_gradients.assign(self.mean_gradients_metric[0])
        tf.summary.scalar(name='fishers/' + self.name, tensor=tf.linalg.norm(self.mean_gradients))

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({'mean_gradient': self.mean_gradients_metric[1],
                                        'assign_mean_gradient': self.assign_mean_gradients,
                                        'global_step': self.global_step})

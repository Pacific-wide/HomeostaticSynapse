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

    opt = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])

    train_op = opt.minimize(loss, global_step=tf.train.get_global_step())

    meta_model = MetaNetwork()

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def multi(features, labels, mode, params):
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

    opt = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])

    train_op = opt.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


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

    checkpoint = params['model_dir']
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

    opt = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])

    grads_and_vars = opt.compute_gradients(loss)
    fisher_hook = []
    for grad_and_var in grads_and_vars:
        fisher_hook.append(GradientAccumulationHook(grad_and_var))

    train_op = opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, training_hooks=fisher_hook, train_op=train_op)


def meta(features, labels, mode, params):
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

    # single 1
    logits1 = model(single_features1)
    one_hot_labels1 = tf.one_hot(single_labels1, 10)
    loss1 = tf.losses.softmax_cross_entropy(one_hot_labels1, logits1)

    # joint
    joint_logits = model(joint_features)
    joint_one_hot_labels = tf.one_hot(joint_labels, 10)
    joint_loss = tf.losses.softmax_cross_entropy(joint_one_hot_labels, joint_logits)

    opt = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])

    grads_and_vars0 = opt.compute_gradients(loss0)
    grads_and_vars1 = opt.compute_gradients(loss1)
    joint_grads_and_vars = opt.compute_gradients(joint_loss)

    grad0, var0 = zip(*grads_and_vars0)
    grad1, var1 = zip(*grads_and_vars1)
    joint_grad, joint_var = zip(*joint_grads_and_vars)

    train_op = opt.apply_gradients(grads_and_vars0, global_step=tf.train.get_global_step())

    meta_model = MetaNetwork()

    meta_batch = combine_meta_batch(grad0, grad1, model.weights)
    meta_label = prepare_meta_batch(joint_grad)
    print("meta_batch", meta_batch)
    print("meta_label", meta_label)

    meta_output = meta_model(meta_batch)
    meta_loss = tf.losses.mean_squared_error(meta_output, meta_label)

    print("meta_model.weights", meta_model.weights)
    meta_grads_and_vars = opt.compute_gradients(meta_loss, var_list=meta_model.weights)

    meta_grad, meta_var = zip(*meta_grads_and_vars)

    meta_opt = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])

    meta_train_op = meta_opt.apply_gradients(meta_grads_and_vars, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=meta_loss, train_op=tf.group([train_op, meta_train_op]))


def meta_eval(features, labels, mode, params):
    model = FullyConnectedNetwork()
    logits = model(features)
    predictions = tf.argmax(logits, axis=1)

    one_hot_labels = tf.one_hot(labels, 10)
    loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels, predictions)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'accuracy': accuracy})

    opt = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
    grads_and_vars = opt.compute_gradients(loss)

    meta_model = MetaNetwork()

    train_op = opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def combine_meta_batch(pre, cur, weight):
    combine_list = [prepare_meta_batch(pre),
                    prepare_meta_batch(cur),
                    prepare_meta_batch(weight)]

    return tf.concat(combine_list, axis=1)


def prepare_meta_batch(grads):
    grad_list = []
    for grad in grads:
        flat_grad = tf.reshape(grad, [-1, 1])
        grad_list.append(flat_grad)

    return tf.concat(grad_list, axis=0)


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
        return self.net(inputs)


class GradientAccumulationHook(tf.train.SessionRunHook):
    def __init__(self, grad_and_var):
        print("Start Accumulation")
        self.gradients = grad_and_var[0]
        self.variable = grad_and_var[1]
        self.name = self.variable.name[:-2]

    def begin(self):
        self.sum_gradients = tf.Variable(tf.zeros_like(self.gradients), name=(self.name + '/fisher'))
        self.global_step = tf.train.get_global_step()
        self.sum_operation = self.sum_gradients.assign_add(tf.math.square(self.gradients))

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({'gradient': self.sum_operation, 'global_step': self.global_step})

    def save_fisher_component(self, results):
        if results['global_step'] % 1000 == 0:
            print(self.name, ': ', np.linalg.norm(results['gradient']))

    def after_run(self, run_context, run_values):
        _ = run_context
        self.save_fisher_component(run_values.results)

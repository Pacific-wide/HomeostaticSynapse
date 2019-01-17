import numpy as np
import tensorflow as tf


def base(features, labels, mode, params):
    model = FullyConnectedNetwork()
    logits= model(features)
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
    logits= model(features)
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

    gradients, variables = zip(*opt.compute_gradients(loss))
    fisher_hook = GradientAccumulationHook(gradients, 2000)

    train_op = opt.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, training_hooks=fisher_hook, train_op=train_op)


class FullyConnectedNetwork(tf.keras.models.Model):
    def __init__(self):
        super(FullyConnectedNetwork, self).__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.InputLayer((784,)),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(10)])

    def call(self, inputs, training=None, mask=None):
        return self.net(inputs)


class GradientAccumulationHook(tf.train.SessionRunHook):
    def __init__(self, gradients, end_step):
        print("Start Accumulation")
        self.sum_gradients = tf.Variable(tf.zeros_like(gradients))
        self.gradients = gradients
        self.end_step = end_step
        self._iter_step = 0

    def begin(self):
        self._iter_step = 0
        if self.iter_step < self.end_step:
            self.sum_gradients.assigned_add(tf.math.square(self.gradients))
        else:
            print("Exit Accumulation")

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self.sum_gradients)

    def save_fisher_component(self, fisher):
        np.save('fisher.npy', fisher)
        print("Start Accumulation")


    def after_run(self, run_context, run_values):
        _ = run_context
        self.save_fisher_component(run_values.results)










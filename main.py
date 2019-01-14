import numpy as np
import tensorflow as tf
import dataset as mnist


def train_input_fn(batch_size):
    dataset_train, _ = mnist.load_mnist_datasets()
    dataset_train = dataset_train.shuffle(1000).repeat(5).batch(batch_size)

    return dataset_train


def eval_input_fn(batch_size):
    _, dataset_eval = mnist.load_mnist_datasets()
    dataset_eval = dataset_eval.batch(batch_size)

    return dataset_eval


def model_fn(features, labels, mode, params):
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

    opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    train_op = opt.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


class FullyConnectedNetwork(tf.keras.models.Model):
    def __init__(self):
        super(FullyConnectedNetwork, self).__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.InputLayer((784,)),
            tf.keras.layers.Dense(800, activation='relu'),
            tf.keras.layers.Dense(800, activation='relu'),
            tf.keras.layers.Dense(10)])

    def call(self, inputs, training=None, mask=None):
        return self.net(inputs)


if __name__ == "__main__":

    batch_size = 64
    run_config = tf.estimator.RunConfig(model_dir='model', save_checkpoints_steps=500)

    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(batch_size=batch_size))
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(batch_size=batch_size),
                                      start_delay_secs=1, throttle_secs=1)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

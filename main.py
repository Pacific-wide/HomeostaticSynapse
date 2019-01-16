from absl import flags
import numpy as np
import tensorflow as tf
import dataset as mnist


# Task flags
flags.DEFINE_integer('n_task', '10', 'Number of tasks in sequence.')
flags.DEFINE_integer('seed', '1', 'Random seed.')

# Estimator flags
flags.DEFINE_string('model_dir', 'model', 'Path to output model directory.')
flags.DEFINE_integer('n_epoch', 1, 'Number of train epochs.')

# Optimizer flags
flags.DEFINE_float('lr', 1e-1, 'Learning rate of an optimizer.')
flags.DEFINE_integer('n_batch', 10, 'Number of examples in a batch')

FLAGS = flags.FLAGS


def prepare_permutations(n_task, seed):
    np.random.seed(seed)
    n_pixel = 784
    p = np.zeros((n_task, n_pixel))
    for i in range(n_task):
        p[i] = np.random.permutation(n_pixel)

    p = p.astype(np.int32)

    return p


def train_input_fn(n_epoch, n_batch, p):
    d_train, _ = mnist.load_mnist_datasets()
    perm_d_train = d_train.map(lambda feature, label: (mnist.permute(feature, p), label))
    perm_d_train = perm_d_train.repeat(n_epoch).batch(n_batch)

    return perm_d_train


def eval_input_fn(n_batch, p):
    _, d_eval = mnist.load_mnist_datasets()
    perm_d_eval = d_eval.map(lambda feature, label: (mnist.permute(feature, p), label))
    perm_d_eval = perm_d_eval.batch(n_batch)

    return perm_d_eval


def model_fn(features, labels, mode, params):
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


def main(unused_argv):
    params = {'learning_rate': FLAGS.lr}
    n_task = FLAGS.n_task
    run_config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=2000)

    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params=params)
    p = prepare_permutations(n_task, FLAGS.seed)

    accuracy_mat = np.zeros((n_task, n_task))
    for i in range(n_task):
        estimator.train(input_fn=lambda: train_input_fn(FLAGS.n_epoch, FLAGS.n_batch, p[i]))
        for j in range(i+1):
            result_dict = estimator.evaluate(input_fn=lambda: eval_input_fn(FLAGS.n_batch, p[j]))
            accuracy_mat[i, j] = result_dict['accuracy']

    print(accuracy_mat)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

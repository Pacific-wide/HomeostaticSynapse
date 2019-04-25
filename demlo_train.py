from absl import flags
import numpy as np
import tensorflow as tf

import input
import model
import dataset as mnist

# Task flag
flags.DEFINE_integer('l_seq', '10', 'Length of a training sequence.')
flags.DEFINE_integer('seed', '2', 'Random seed.')

# Estimator flags
flags.DEFINE_string('model_dir', 'model', 'Path to output model directory.')
flags.DEFINE_integer('n_epoch', 1, 'Number of train epochs.')

# Optimizer flags
flags.DEFINE_float('lr', 5e-2, 'Learning rate of an main optimizer.')
flags.DEFINE_float('meta_lr', 1e-3, 'Learning rate of an meta optimizer.')
flags.DEFINE_integer('n_batch', 10, 'Number of examples in a batch')

FLAGS = flags.FLAGS


def prepare_permutations(n_task, seed):
    np.random.seed(seed)
    n_pixel = 784
    n_task_ext = 2 * n_task - 1
    p = np.zeros((n_task_ext, n_pixel))
    for i in range(n_task):
        p[i] = np.random.permutation(n_pixel)
        if i < (n_task-1):
            p[n_task_ext-i-1] = p[i]

    p = p.astype(np.int32)

    return p


def main(unused_argv):
    run_config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=12000)
    params = {'lr': FLAGS.lr, 'meta_lr': FLAGS.meta_lr, 'layers': [20, 20], 'model': FLAGS.model_dir}

    l_seq = FLAGS.l_seq
    p = prepare_permutations(l_seq, FLAGS.seed)
    x_tr, y_tr, x_te, y_te = mnist.load_mnist_datasets()

    pre_estimator = tf.estimator.Estimator(model_fn=model.meta_base, config=run_config, params=params)

    # 1st Task SGD scratch learning
    pre_estimator.train(input_fn=lambda: input.train_input_fn(x_tr, y_tr, FLAGS.n_epoch, FLAGS.n_batch, p[0]))

    # Meta network training

    for i in range(1, l_seq):
        estimator = tf.estimator.Estimator(model_fn=model.meta_ewc,
                                           config=run_config,
                                           params=params)
        estimator.train(input_fn=lambda: input.train_input_fn(x_tr, y_tr, FLAGS.n_epoch, FLAGS.n_batch, p[i]))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

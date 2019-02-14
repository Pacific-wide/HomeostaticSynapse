from absl import flags
import numpy as np
import tensorflow as tf

import input
import model

# Task flag
flags.DEFINE_integer('n_seq', '20', 'Number of training sequences.')
flags.DEFINE_integer('seed', '1', 'Random seed.')

# Estimator flags
flags.DEFINE_string('model', 'meta_train', 'Model to train.')
flags.DEFINE_string('model_dir', 'model', 'Path to output model directory.')
flags.DEFINE_integer('n_epoch', 4, 'Number of train epochs.')

# Optimizer flags
flags.DEFINE_float('lr', 1e-3, 'Learning rate of an main optimizer.')
flags.DEFINE_float('meta_lr', 1e-4, 'Learning rate of an  optimizer.')
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


def main(unused_argv):
    run_config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
                                        save_checkpoints_steps=2000)

    params = {'lr': FLAGS.lr, 'meta_lr': FLAGS.meta_lr,
              'model_dir': FLAGS.model_dir,
              'layers': [784, 100, 100, 10]}
    model_dict = {'base': model.base,
                  'ewc': model.ewc,
                  'meta_train': model.meta_train,
                  'meta_eval': model.meta_eval}

    n_seq = FLAGS.n_seq

    estimator = tf.estimator.Estimator(model_fn=model_dict[FLAGS.model],
                                       config=run_config,
                                       params=params)

    for t in range(n_seq):
        p = prepare_permutations(10, FLAGS.seed)
        for i in range(9):
            cur_p = p[i:i + 2]
            estimator.train(input_fn=lambda: input.meta_train_input_fn(FLAGS.n_epoch, FLAGS.n_batch, cur_p))
        print('-' * 50 + "Seq " + str(t + 1) + " Complete " + '-' * 50)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

from absl import flags
import numpy as np
import tensorflow as tf

import input
import model
import dataset as mnist

# Task flag
flags.DEFINE_integer('l_seq', '5', 'Length of a training sequence.')
flags.DEFINE_integer('seed', '10', 'Random seed.')
flags.DEFINE_integer('k_rep', '10', 'Repeat number of training sequence.')

# Estimator flags
flags.DEFINE_string('model_dir', 'model', 'Path to output model directory.')
flags.DEFINE_integer('n_epoch', 1, 'Number of train epochs.')

# Optimizer flags
flags.DEFINE_float('lr', 5e-2, 'Learning rate of an main optimizer.')
flags.DEFINE_float('meta_lr', 1e-3, 'Learning rate of an meta optimizer.')
flags.DEFINE_integer('n_batch', 10, 'Number of examples in a batch')

FLAGS = flags.FLAGS


def make_matching_dict():
    matching_dict = {}
    layers = ['dense1', 'dense2', 'dense3']
    kinds = ['kernel', 'bias']

    for layer in layers:
        for kind in kinds:
            sur =  layer + '/' + kind
            key = 'pre/' + sur
            value = 'main/' + sur
            fisher_key = 'fisher/' + sur
            fisher_value = 'main/' + sur + '/fisher'
            matching_dict[key] = value
            matching_dict[fisher_key] = fisher_value

    return matching_dict


def prepare_permutations(n_task, seed, k):
    np.random.seed(seed)
    n_pixel = 784
    n_task_ext = k * n_task
    p = np.zeros((n_task_ext, n_pixel))
    for i in range(n_task):
        p[i] = np.random.permutation(n_pixel)
        for j in range(k-1):
            p[i + n_task*(j+1)] = p[i]

    p = p.astype(np.int32)

    return p


def main(unused_argv):
    run_config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=3000)
    params = {'lr': FLAGS.lr, 'meta_lr': FLAGS.meta_lr, 'layers': [20, 20]}

    l_seq = FLAGS.l_seq
    l_seq_ext = FLAGS.k_rep * l_seq

    p = prepare_permutations(l_seq, FLAGS.seed, FLAGS.k_rep)

    x_tr, y_tr, x_te, y_te = mnist.load_mnist_datasets()

    pre_estimator = tf.estimator.Estimator(model_fn=model.base, config=run_config, params=params)

    # 1st Task SGD scratch learning
    pre_estimator.train(input_fn=lambda: input.train_input_fn(x_tr, y_tr, FLAGS.n_epoch, FLAGS.n_batch, p[0]))

    # Meta network training
    matching_dict = make_matching_dict()

    for i in range(1, l_seq_ext):
        ws = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=tf.train.latest_checkpoint('model'),
            var_name_to_prev_var_name=matching_dict)
        print(tf.train.latest_checkpoint('model'))
        estimator = tf.estimator.Estimator(model_fn=model.meta_ewc,
                                           config=run_config,
                                           params=params, warm_start_from=ws)
        estimator.train(input_fn=lambda: input.train_input_fn(x_tr, y_tr, FLAGS.n_epoch, FLAGS.n_batch, p[i]))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

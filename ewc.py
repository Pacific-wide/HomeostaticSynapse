from absl import flags
import numpy as np
import tensorflow as tf

import input
import model
import os
import dataset as data


# Task flag
flags.DEFINE_integer('seed', '1', 'Random seed.')
flags.DEFINE_integer('n_task_eval', '10', 'Number of tasks in evaluation sequence.')

# Estimator flag
flags.DEFINE_string('model', 'ewc', 'Model to evaluate.')
flags.DEFINE_string('eval_model_dir', 'eval_model', 'Path to test model directory.')
flags.DEFINE_integer('n_epoch', 1, 'Number of train epochs.')

# Optimizer flags
flags.DEFINE_float('lr', 5e-2, 'Learning rate of an main optimizer.')
flags.DEFINE_float('alpha', 3.0, 'alpha for additional loss')
flags.DEFINE_integer('n_batch', 10, 'Number of examples in a batch')

FLAGS = flags.FLAGS


def save_confusion_matrix(conf_mat, cur_model):
    path = 'result/'
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + cur_model + '.npy', conf_mat)


def prepare_permutations(n_task, seed):
    np.random.seed(seed)
    n_pixel = 784
    p = np.zeros((n_task, n_pixel))
    for i in range(n_task):
        p[i] = np.random.permutation(n_pixel)

    p = p.astype(np.int32)

    return p


def main(unused_argv):
    eval_run_config = tf.estimator.RunConfig(model_dir=FLAGS.eval_model_dir,
                                             save_checkpoints_steps=6000)

    params = {'lr': FLAGS.lr,
              'eval_model_dir': FLAGS.eval_model_dir,
              'layers': [784, 100, 100, 10],
              'alpha': FLAGS.alpha}

    pre_estimator = tf.estimator.Estimator(model_fn=model.base,
                                           config=eval_run_config,
                                           params=params)
    estimator = tf.estimator.Estimator(model_fn=model.ewc,
                                       config=eval_run_config,
                                       params=params)

    n_task_eval = FLAGS.n_task_eval
    p_eval = prepare_permutations(n_task_eval, FLAGS.seed)   # permutation maps

    x_tr, y_tr, x_te, y_te = data.load_mnist_datasets()

    eval_accuracy_mat = np.zeros((n_task_eval, n_task_eval))

    # 1st Task SGD scratch learning
    pre_estimator.train(input_fn=lambda: input.train_input_fn(x_tr, y_tr, FLAGS.n_epoch, FLAGS.n_batch, p_eval[0]))
    result_dict = pre_estimator.evaluate(input_fn=lambda: input.eval_input_fn(x_te, y_te, FLAGS.n_batch, p_eval[0]))
    eval_accuracy_mat[0, 0] = result_dict['accuracy']

    # 2nd to nth Tasks learning with Meta network (new optimizer)
    for i in range(1, n_task_eval):
        estimator.train(input_fn=lambda: input.train_input_fn(x_tr, y_tr, FLAGS.n_epoch, FLAGS.n_batch, p_eval[i]))

        for j in range(i+1):
            result_dict = estimator.evaluate(input_fn=lambda: input.eval_input_fn(x_te, y_te, FLAGS.n_batch, p_eval[j]))
            eval_accuracy_mat[i, j] = result_dict['accuracy']
            np.set_printoptions(precision=4)

    print(eval_accuracy_mat)
    save_confusion_matrix(eval_accuracy_mat, FLAGS.model)
    print(FLAGS.model)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

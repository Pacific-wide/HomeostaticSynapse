from absl import flags
import numpy as np
import tensorflow as tf

import input
import model
import plot
import os

# Task flag
flags.DEFINE_integer('n_seq', '40', 'Number of training sequences.')
flags.DEFINE_integer('n_task', '2', 'Number of tasks in training sequence.')
flags.DEFINE_integer('seed', '1', 'Random seed.')
flags.DEFINE_integer('n_task_eval', '10', 'Number of tasks in evaluation sequence.')

# Estimator flag
flags.DEFINE_string('model', 'meta', 'Model to evaluate.')
flags.DEFINE_string('model_dir', 'model', 'Path to output model directory.')
flags.DEFINE_string('eval_model_dir', 'eval_model', 'Path to test model directory.')
flags.DEFINE_integer('n_epoch', 4, 'Number of train epochs.')

# Optimizer flags
flags.DEFINE_float('lr', 1e-3, 'Learning rate of an main optimizer.')
flags.DEFINE_float('meta_lr', 1e-4, 'Learning rate of an  optimizer.')
flags.DEFINE_integer('n_batch', 10, 'Number of examples in a batch')

# Meta-Learning flags

FLAGS = flags.FLAGS

def save_confusion_matrix(conf_mat, model):
    path = 'result/'
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + model + 'npy', conf_mat)


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
    eval_run_config = tf.estimator.RunConfig(model_dir=FLAGS.eval_model_dir,
                                             save_checkpoints_steps=6000)

    params = {'lr': FLAGS.lr, 'meta_lr': FLAGS.meta_lr,
              'model_dir': FLAGS.model_dir,
              'eval_model_dir': FLAGS.eval_model_dir,
              'layers': [784, 100, 100, 10]}
    model_dict = {'base': model.base,
                  'ewc': model.ewc,
                  'meta_train': model.meta_train,
                  'meta': model.meta_eval}

    meta_matching_dict = {"dense_3/bias": "dense_3/bias",
                          "dense_3/kernel": "dense_3/kernel",
                          "dense_4/bias": "dense_4/bias",
                          "dense_4/kernel": "dense_4/kernel",
                          "dense_5/bias": "dense_5/bias",
                          "dense_5/kernel": "dense_5/kernel"}

    hook_matching_dict = {**meta_matching_dict}

    ws0 = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=tf.train.latest_checkpoint('/tmp/pycharm_project_91/model'),
                                         var_name_to_prev_var_name=meta_matching_dict)

    estimator0 = tf.estimator.Estimator(model_fn=model.base,
                                        config=eval_run_config,
                                        params=params, warm_start_from=ws0)

    estimator1 = tf.estimator.Estimator(model_fn=model_dict[FLAGS.model],
                                        config=eval_run_config,
                                        params=params)
    n_task_eval = FLAGS.n_task_eval
    p_eval = prepare_permutations(n_task_eval, FLAGS.seed+10)   # permutation maps

    eval_accuracy_mat = np.zeros((n_task_eval, n_task_eval))

    # 1st Task SGD scratch learning
    estimator0.train(input_fn=lambda: input.train_input_fn(FLAGS.n_epoch, FLAGS.n_batch, p_eval[0]), max_steps=6000)
    result_dict = estimator0.evaluate(input_fn=lambda: input.eval_input_fn(FLAGS.n_batch, p_eval[0]))
    eval_accuracy_mat[0, 0] = result_dict['accuracy']

    # 2nd to nth Tasks learning with Meta network (new opimizer)
    for i in range(1, n_task_eval):
        estimator1.train(input_fn=lambda: input.train_input_fn(FLAGS.n_epoch, FLAGS.n_batch, p_eval[i]), max_steps=6000*(i+1))
        for j in range(i+1):
            result_dict = estimator1.evaluate(input_fn=lambda: input.eval_input_fn(FLAGS.n_batch, p_eval[j]))
            eval_accuracy_mat[i, j] = result_dict['accuracy']
            np.set_printoptions(precision=4)
            print(eval_accuracy_mat)
        print('-'*50 + "Task " + str(i) + " Complete " + '-'*50)

    plot.save_confusion_matrix(eval_accuracy_mat, FLAGS.model)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

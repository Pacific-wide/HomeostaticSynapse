from absl import flags
import numpy as np
import tensorflow as tf

import input
import model

# Task flag
flags.DEFINE_integer('n_seq', '10', 'Number of sequences.')
flags.DEFINE_integer('n_task', '2', 'Number of tasks in sequence.')
flags.DEFINE_integer('seed', '1', 'Random seed.')
flags.DEFINE_integer('n_task_eval', '10', 'Number of tasks in evaluation sequence.')

# Estimator flags
flags.DEFINE_string('model', 'meta', 'Model to train.')
flags.DEFINE_string('model_dir', 'model', 'Path to output model directory.')
flags.DEFINE_string('test_model_dir', 'test_model', 'Path to test model directory.')
flags.DEFINE_integer('n_epoch', 1, 'Number of train epochs.')

# Optimizer flags
flags.DEFINE_float('lr', 1e-1, 'Learning rate of an optimizer.')
flags.DEFINE_integer('n_batch', 10, 'Number of examples in a batch')

# Meta-Learning flags

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
    eval_run_config = tf.estimator.RunConfig(model_dir=FLAGS.test_model_dir,
                                        save_checkpoints_steps=6000)

    params = {'learning_rate': FLAGS.lr, 'model_dir': FLAGS.model_dir,
              'layers': [784, 100, 100, 10]}
    model_dict = {'base': model.base,
                  'multi': model.multi,
                  'ewc': model.ewc,
                  'meta': model.meta}
    n_task = FLAGS.n_task
    n_seq = FLAGS.n_seq

    ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from="/tmp/m")

    estimator0 = tf.estimator.Estimator(model_fn=model.base,
                                       config=eval_run_config,
                                       params=params)
    estimator1 = tf.estimator.Estimator(model_fn=model.meta_eval,
                                       config=eval_run_config,
                                       params=params)
    n_task_eval = FLAGS.n_task_eval
    p_eval = prepare_permutations(n_task_eval, FLAGS.seed+10)

    eval_accuracy_mat = np.zeros((n_task_eval, n_task_eval))

    estimator0.train(input_fn=lambda: input.train_input_fn(FLAGS.n_epoch, FLAGS.n_batch, p_eval[0]), max_steps=6000)
    result_dict = estimator0.evaluate(input_fn=lambda: input.eval_input_fn(FLAGS.n_batch, p_eval[i]))
    eval_accuracy_mat[0, 0] = result_dict['accuracy']

    for i in range(1, n_task_eval):
        estimator1.train(input_fn=lambda: input.train_input_fn(FLAGS.n_epoch, FLAGS.n_batch, p_eval[i]), max_steps=6000)
        for j in range(i+1):
            result_dict = estimator0.evaluate(input_fn=lambda: input.eval_input_fn(FLAGS.n_batch, p_eval[i]))
            eval_accuracy_mat[i, j] = result_dict['accuracy']
        print('-'*50 + "Task " + str(i) + " Complete " + '-'*50)
        
    np.set_printoptions(precision=4)
    print(eval_accuracy_mat)

    '''
        estimator = tf.estimator.Estimator(model_fn=model_dict[FLAGS.model],
                                           config=run_config,
                                           params=params)

        for t in range(n_seq):
            p = prepare_permutations(n_task, FLAGS.seed)
            estimator.train(input_fn=lambda: input.meta_train_input_fn(FLAGS.n_epoch, FLAGS.n_batch, p))
            accuracy_mat = np.zeros(n_task)
            for i in range(n_task):
                result_dict = estimator.evaluate(input_fn=lambda: input.eval_input_fn(FLAGS.n_batch, p[i]))
                accuracy_mat[i] = result_dict['accuracy']
                print('-' * 50 + "Task " + str(i + 1) + " Complete " + '-' * 50)
            print('-' * 50 + "Seq " + str(t + 1) + " Complete " + '-' * 50)
            np.set_printoptions(precision=4)
            print(accuracy_mat)
    '''



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

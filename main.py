from absl import flags
import numpy as np
import tensorflow as tf

import input
import model

# Task flags
flags.DEFINE_integer('n_task', '5', 'Number of tasks in sequence.')
flags.DEFINE_integer('seed', '1', 'Random seed.')

# Estimator flags
flags.DEFINE_string('model', 'multi', 'Model to train.')
flags.DEFINE_string('model_dir', 'model', 'Path to output model directory.')
flags.DEFINE_integer('n_epoch', 5, 'Number of train epochs.')

# Optimizer flags
flags.DEFINE_float('lr', 1e-2, 'Learning rate of an optimizer.')
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

    params = {'learning_rate': FLAGS.lr, 'model_dir': FLAGS.model_dir}
    model_dict = {'base': model.base,
                  'multi': model.multi,
                  'ewc': model.ewc}
    n_task = FLAGS.n_task
    p = prepare_permutations(n_task, FLAGS.seed)

    estimator = tf.estimator.Estimator(model_fn=model_dict[FLAGS.model],
                                       config=run_config,
                                       params=params)

    accuracy_mat = np.zeros((n_task, n_task))
    for i in range(n_task):
        estimator.train(input_fn=lambda: input.train_input_fn(FLAGS.n_epoch, FLAGS.n_batch, p[i]))
        for j in range(i+1):
            result_dict = estimator.evaluate(input_fn=lambda: input.eval_input_fn(FLAGS.n_batch, p[j]))
            accuracy_mat[i, j] = result_dict['accuracy']
        print('-'*50 + "Task " + str(i) + " Complete " + '-'*50)
        
    np.set_printoptions(precision=4)
    print(accuracy_mat)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

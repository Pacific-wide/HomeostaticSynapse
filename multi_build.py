import dataset
import tensorflow as tf
import learner
import optimizer as op
import numpy as np


def main(unused_argv):
    learning_rate = 5e-2
    n_epoch = 1
    n_task = 10

    np.random.seed(2)
    n_batch = n_task * 10

    run_config = tf.estimator.RunConfig(model_dir="multi"+str(n_task), save_checkpoints_steps=6000)

    set_single_dataset = dataset.SetOfRandPermMnist(n_task)

    my_opt = op.SGDOptimizer().build(learning_rate)
    my_opt_spec = learner.OptimizerSpec(my_opt, learning_rate)
    my_learning_spec = learner.LearningSpec(n_epoch, n_batch, n_task, my_opt_spec)

    multi_dataset = set_single_dataset.concat()
    multi_learner = learner.MultiEstimatorLearner(multi_dataset, my_learning_spec, run_config)
    multi_learner.train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

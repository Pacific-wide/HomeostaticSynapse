import dataset
import tensorflow as tf
import learner
import optimizer as op
import numpy as np


def main(unused_argv):
    learning_rate = 5e-2
    n_epoch = 1
    n_batch = 10
    n_task = 10
    np.random.seed(2)

    run_config = tf.estimator.RunConfig(model_dir="meta", save_checkpoints_steps=6000)

    set_meta_dataset = dataset.SetOfRandPermMnist(n_task)

    my_opt = op.SGDOptimizer().build(learning_rate)
    my_opt_spec = learner.OptimizerSpec(my_opt, learning_rate)
    my_learning_spec = learner.LearningSpec(n_epoch, n_batch, my_opt_spec)

    for i in range(n_task):
        meta_dataset = set_meta_dataset.list[i]
        meta_learner = learner.MetaEstimatorLearner(meta_dataset, my_learning_spec, run_config)
        meta_learner.train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

import dataset
import tensorflow as tf
import grouplearner
import spec
import optimizer as op
import numpy as np


def main(unused_argv):
    learning_rate = 5e-2
    n_epoch = 1
    n_batch = 10
    n_task = 10
    model_dir = "single"

    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=6000)

    set_of_datasets = dataset.SetOfRandPermMnist(n_task)

    d_in = set_of_datasets.list[0].d_in

    opt = op.SGDOptimizer().build(learning_rate)
    opt_spec = spec.OptimizerSpec(opt, d_in)
    learning_spec = spec.LearningSpec(n_epoch, n_batch, n_task, model_dir, opt_spec)

    my_grouplearner = grouplearner.GroupSingleLearner(set_of_datasets, learning_spec, n_task, run_config)

    accuracy_matrix = my_grouplearner.train_and_evaluate()

    np.set_printoptions(precision=4)
    print(accuracy_matrix)

    n_total = n_task*(n_task+1) / 2.0
    print(accuracy_matrix.sum()/n_total)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

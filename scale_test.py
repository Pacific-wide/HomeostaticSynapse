import dataset
import tensorflow as tf
import grouplearner
import spec
import optimizer as op
import numpy as np
import sys


def main(argv):
    learning_rates = [5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-4, 5e-4, 5e-4, 5e-4, 5e-4]
    learning_specs = []
    n_epoch = 1
    n_batch = 10
    n_task = len(learning_rates)
    model_dir = "scale"
    np.random.seed(0)

    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=6000)

    set_of_datasets = dataset.SetOfRandPermMnist(n_task)

    d_in = set_of_datasets.list[0].d_in

    for i in range(n_task):
        opt = op.SGDOptimizer().build(learning_rates[i])
        opt_spec = spec.OptimizerSpec(opt, d_in)
        learning_specs.append(spec.LearningSpec(n_epoch, n_batch, n_task, model_dir, opt_spec))

    my_grouplearner = grouplearner.GroupScaleLearner(set_of_datasets, learning_specs[0], n_task, run_config, learning_specs)

    accuracy_matrix = my_grouplearner.train_and_evaluate()

    np.set_printoptions(precision=4)

    n_total = n_task*(n_task+1) / 2.0
    average_accuracy = accuracy_matrix.sum()/n_total

    print(accuracy_matrix)
    print(average_accuracy)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

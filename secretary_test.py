import dataset
import tensorflow as tf
import grouplearner
import spec
import optimizer as op
import numpy as np
import sys


def main(argv):
    print(argv)
    alpha = float(argv[1])
    skip_rate = float(argv[2])
    lr = float(argv[3])

    learning_rate = lr
    n_epoch = 1
    n_batch = 10
    n_task = 20
    learning_rates = learning_rate * np.ones(n_task)
    learning_specs = []

    model_dir = "sec"
    np.random.seed(0)

    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=6000)

    set_of_datasets = dataset.SetOfRandPermMnist(n_task)

    d_in = set_of_datasets.list[0].d_in

    for i in range(n_task):
        opt = op.SGDOptimizer().build(learning_rates[i])
        opt_spec = spec.OptimizerSpec(opt, d_in)
        learning_specs.append(spec.LearningSpec(n_epoch, n_batch, n_task, model_dir, opt_spec, alpha, skip_rate))

    my_grouplearner = grouplearner.GroupREWCLearner(set_of_datasets, learning_specs, n_task, run_config)

    accuracy_matrix = my_grouplearner.train_and_evaluate()

    np.set_printoptions(precision=4)

    n_total = n_task*(n_task+1) / 2.0
    average_accuracy = accuracy_matrix.sum()/n_total

    print(accuracy_matrix)
    print(average_accuracy)

    filepath = "sec_result.txt"
    f = open(filepath, 'a')
    f.write("(alpha, skip_rate, lr) = (" + str(alpha) +","+ str(skip_rate) +","+ str(lr) + ") ")
    f.write(str(round(average_accuracy, 4)) + "\n")
    f.close()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

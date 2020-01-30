import dataset
import tensorflow as tf
import grouplearner
import spec
import optimizer as op
import numpy as np
import metric
import logger
import sys


def main(argv):
    print(argv)
    alpha = float(argv[1])
    learning_rate = float(argv[2])

    learning_rate = 5e-2
    n_epoch = 1
    n_batch = 100
    n_task = int(argv[1])
    alpha = 0.1
    learning_rates = learning_rate * np.ones(n_task)
    learning_specs = []

    model_dir = "multi"+str(n_task)
    np.random.seed(2)

    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=6000)

    set_of_datasets = dataset.SetOfRandPermMnist(n_task)

    d_in = set_of_datasets.list[0].d_in

    for i in range(n_task):
        opt = op.SGDOptimizer().build(learning_rates[i])
        opt_spec = spec.OptimizerSpec(opt, d_in)
        learning_specs.append(spec.LearningSpec(n_epoch, n_batch, n_task, model_dir, opt_spec, alpha))

    my_grouplearner = grouplearner.GroupSingleLearner(set_of_datasets, learning_specs, n_task, run_config)

    accuracy_matrix = my_grouplearner.train_and_evaluate()

    avg_acc = metric.AverageAccuracy(accuracy_matrix).compute()
    tot_acc = metric.TotalAccuracy(accuracy_matrix).compute()
    avg_forget = metric.AverageForgetting(accuracy_matrix).compute()
    tot_forget = metric.TotalForgetting(accuracy_matrix).compute()

    metric_list = [avg_acc, tot_acc, avg_forget, tot_forget]

    filepath = "r_multi.txt"
    logger.save(filepath, accuracy_matrix, metric_list, seed, learning_specs)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

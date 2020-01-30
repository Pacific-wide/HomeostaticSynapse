import dataset
import tensorflow as tf
import grouplearner
import optimizer as op
import spec
import metric
import logger
import numpy as np


def main(argv):
    print(argv)
    learning_rate = 1e-1
    n_epoch = 1
    seed = int(argv[1])
    alpha = 0
    n_task = 10
    n_batch = 100
    n_multibatch = n_task * n_batch
    model_dir = "multi"

    learning_rates = learning_rate * np.ones(n_task)
    learning_specs = []

    np.random.seed(seed)

    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=600)

    set_of_datasets = dataset.SetOfRandPermMnist(n_task)
    d_in = set_of_datasets.list[0].d_in

    for i in range(n_task):
        opt = op.SGDOptimizer().build(learning_rates[i])
        opt_spec = spec.OptimizerSpec(opt, d_in)
        learning_specs.append(spec.LearningSpec(n_epoch, n_multibatch, n_task, model_dir, opt_spec, alpha))

    multi_task_grouplearner = grouplearner.GroupMultiLearner(set_of_datasets, learning_specs, n_task, run_config)

    accuracy_vector = multi_task_grouplearner.train_and_evaluate()

    filepath = "r_multi.txt"
    avg_acc = accuracy_vector.sum() / (1.0 * n_task)
    metric_list = [avg_acc]
    accuracy_vector = accuracy_vector.reshape(1,-1)
    logger.save(filepath, accuracy_vector, metric_list, seed, learning_specs)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

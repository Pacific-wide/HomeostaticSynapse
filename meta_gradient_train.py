import tensorflow as tf
import numpy as np

from dataset import dataset
from model import grouplearner
from optimizer import optimizer as op
from optimizer import spec


def main(unused_argv):
    # learning rate
    learning_rate = 5e-2
    meta_learning_rate = 5e-4

    # learning parameter
    n_epoch = 1
    n_task = 30
    n_batch = 100
    learning_rates = learning_rate * np.ones(n_task)
    learning_specs = []

    # model path
    model_dir = "meta_gra"

    np.random.seed(20)

    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=6000)

    # generate sequence dataset
    set_of_datasets = dataset.SetOfRandPermMnist(n_task + 1)
    d_in = set_of_datasets.list[0].d_in

    for i in range(n_task):
        opt = op.SGDOptimizer().build(learning_rates[i])
        opt_spec = spec.OptimizerSpec(opt, d_in)
        learning_specs.append(spec.LearningSpec(n_epoch, n_batch, n_task, model_dir, opt_spec))

    meta_opt = op.SGDOptimizer().build(meta_learning_rate)
    meta_opt_spec = spec.OptimizerSpec(meta_opt, d_in)
    meta_learning_spec = spec.LearningSpec(n_epoch, n_batch, n_task, model_dir, meta_opt_spec)

    my_grouplearner = grouplearner.GroupMetaGradientTrainLearner(set_of_datasets, learning_specs, n_task, run_config, meta_learning_spec)
    my_grouplearner.train_and_evaluate()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

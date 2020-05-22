import tensorflow as tf
import numpy as np

from dataset import set_of_dataset as sod
from model import grouplearner
from optimizer import optimizer as op
from optimizer import spec


def main(argv):
    # learning rate
    learning_rate = 5e-2
    meta_learning_rate = 5e-4
    seed = int(argv[1])
    step = int(argv[2])
    # learning parameter
    n_epoch = 1
    n_task = 20
    n_batch = 100
    learning_rates = learning_rate * np.ones(n_task)
    learning_specs = []
    n_grid = 7

    np.random.seed(seed)
    # model path
    model_dir = "meta_alpha_ep0"

    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=int(60000/n_batch))

    # generate sequence dataset
    set_of_datasets = sod.SetOfSwapGrid(n_task+1, n_grid, step=step)
    d_in = set_of_datasets.list[0].d_in

    for i in range(n_task):
        opt = op.SGDOptimizer(learning_rates[i]).build()
        opt_spec = spec.OptimizerSpec(opt, d_in)
        learning_specs.append(spec.LearningSpec(n_epoch, n_batch, n_task, model_dir, opt_spec))

    meta_opt = op.SGDOptimizer(meta_learning_rate).build()
    meta_opt_spec = spec.OptimizerSpec(meta_opt, d_in)
    meta_learning_spec = spec.LearningSpec(n_epoch, n_batch, n_task, model_dir, meta_opt_spec)

    joint_opt = op.SGDOptimizer(0).build()
    joint_opt_spec = spec.OptimizerSpec(joint_opt, d_in)
    joint_learning_spec = spec.LearningSpec(1, set_of_datasets.list[0].n_train, 2, model_dir, joint_opt_spec)

    my_grouplearner = grouplearner.GroupMetaAlphaTrainLearner(set_of_datasets, learning_specs, n_task, run_config,
                                                              meta_learning_spec,
                                                              joint_learning_spec)
    my_grouplearner.train_and_evaluate()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

import dataset
import tensorflow as tf
import grouplearner
import spec
import optimizer as op
import numpy as np


def main(unused_argv):
    # learning rate
    learning_rate = 5e-2
    meta_learning_rate = 5e-4

    # learning parameter
    n_epoch = 1
    n_task = 10
    n_batch = 10

    # model path
    model_dir = "meta"

    np.random.seed(2)

    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=6000)
    ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=model_dir, vars_to_warm_start="main*")

    # generate sequence dataset
    set_of_datasets = dataset.SetOfRandPermMnist(n_task + 1)
    d_in = set_of_datasets.list[0].d_in

    opt = op.SGDOptimizer().build(learning_rate)
    opt_spec = spec.OptimizerSpec(opt, d_in)
    learning_spec = spec.LearningSpec(n_epoch, n_batch, n_task, model_dir, opt_spec)

    meta_opt = op.SGDOptimizer().build(meta_learning_rate)
    meta_opt_spec = spec.OptimizerSpec(meta_opt, d_in)
    meta_learning_spec = spec.LearningSpec(n_epoch, n_batch, n_task, model_dir, meta_opt_spec)

    my_grouplearner = grouplearner.GroupMetaTrainLearner(set_of_datasets, learning_spec, n_task, run_config, meta_learning_spec)
    my_grouplearner.train_and_evaluate()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

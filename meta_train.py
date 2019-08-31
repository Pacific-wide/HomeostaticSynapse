import dataset
import tensorflow as tf
import learner
import optimizer as op
import numpy as np


def main(unused_argv):
    # learning rate
    learning_rate = 5e-3
    meta_learning_rate = 5e-3

    # learning parameter
    n_epoch = 1
    n_task = 10
    n_batch = 10

    # model path
    first_model_dir = "first"
    model_dir = "meta"
    pre_model_dir = model_dir

    np.random.seed(5)

    first_run_config = tf.estimator.RunConfig(model_dir=first_model_dir, save_checkpoints_steps=6000)
    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=6000)

    # learning specs
    opt = op.SGDOptimizer().build(learning_rate)
    opt_spec = learner.OptimizerSpec(opt, learning_rate, pre_model_dir)
    learning_spec = learner.LearningSpec(n_epoch, n_batch, n_task, model_dir, opt_spec)

    meta_opt = op.SGDOptimizer().build(meta_learning_rate)
    meta_opt_spec = learner.OptimizerSpec(meta_opt, meta_learning_rate, pre_model_dir)
    meta_learning_spec = learner.LearningSpec(n_epoch, n_batch, n_task, model_dir, meta_opt_spec)

    base_opt_spec = learner.OptimizerSpec(meta_opt, meta_learning_rate, first_model_dir)
    base_learning_spec = learner.LearningSpec(n_epoch, n_batch, n_task, model_dir, base_opt_spec)

    # generate sequence dataset
    set_single_dataset = dataset.SetOfRandPermMnist(n_task+1)

    # Base training
    sinlge_dataset = set_single_dataset.list[n_task]
    single_learner = learner.SingleEstimatorLearner(sinlge_dataset, learning_spec, first_run_config)
    single_learner.train()

    meta_dataset0 = set_single_dataset.list[0:2]
    meta_learner0 = learner.MetaEstimatorLearner(meta_dataset0, base_learning_spec, meta_learning_spec, run_config)
    meta_learner0.train()

    # Sequence training
    for i in range(1, n_task-1):
        meta_dataset = set_single_dataset.list[i:i+2]
        meta_learner = learner.MetaEstimatorLearner(meta_dataset, learning_spec, meta_learning_spec, run_config)
        meta_learner.train()
        print('-' * 50 + "Task (" + str(i) + "," + str(i + 1) + ") Complete " + '-' * 50)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

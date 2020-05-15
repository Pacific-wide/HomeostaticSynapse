import tensorflow as tf
import numpy as np

from dataset import set_of_dataset as sod
from model import learner
from optimizer import optimizer as op
from optimizer import spec


def main(argv):
    n_task = 2
    n_grid = 7
    n_batch = 10
    model_dir = "test"
    np.random.seed(0)

    set_of_datasets = sod.SetOfRandGridPermMnist(n_task, n_grid)
    my_dataset = set_of_datasets.concat()
    n_train = set_of_datasets.list[0].n_train
    d_in = set_of_datasets.list[0].d_in

    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=int(n_train/n_batch))

    joint_opt = op.SGDOptimizer(0).build()
    joint_opt_spec = spec.OptimizerSpec(joint_opt, d_in)
    joint_learning_spec = spec.LearningSpec(1, n_train, 2, model_dir, joint_opt_spec)

    my_learner = learner.JointEstimatorLearner(my_dataset, joint_learning_spec, run_config)
    my_learner.train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

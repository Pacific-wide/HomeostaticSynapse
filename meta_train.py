import tensorflow as tf
import numpy as np
import argparse
import importlib

from model import grouplearner
from optimizer import optimizer as op
from optimizer import spec


def main(argv):
    parser = argparse.ArgumentParser(description='Homeostatic Synapse')

    # model parameters
    parser.add_argument('--model', type=str, default='Single', help='main learner')
    parser.add_argument('--alpha', type=float, default=1.0, help='Intensity of Regularization')

    # data parameters
    parser.add_argument('--data', type=str, default='MNISTPERM', help='Type of Dataset')

    # optimizer parameters
    parser.add_argument('--n_epoch', type=int, default=1, help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-2, help='SGD learning rate for main network')
    parser.add_argument('--meta_lr', type=float, default=5e-2, help='SGD learning rate for HM')

    # experiment parameters
    parser.add_argument('--n_task', type=int, default=10, help='Number of tasks')
    parser.add_argument('--n_block', type=int, default=7, help='Number of blocks in BPERM')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--save_path', type=str, default='results/', help='save models')

    args = parser.parse_args()

    seed = args.seed
    alpha = args.alpha
    learning_rate = args.lr
    meta_learning_rate = args.meta_lr
    n_epoch = args.n_epoch
    n_batch = args.batch_size
    n_task = args.n_task
    n_block = args.n_block
    np.random.seed(seed)

    model_dir = 'HMTrain'

    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=int(60000/n_batch))
    DataClass = getattr(importlib.import_module('dataset.set_of_dataset'), 'SetOfRand' + args.data)

    # generate sequence dataset
    if args.data[-5:] == 'BPERM':
        set_of_datasets = DataClass(n_task+1, n_block)        # For Block-wise Permutation
    else:
        set_of_datasets = DataClass(n_task+1)

    d_in = set_of_datasets.list[0].d_in
    n_train = set_of_datasets.list[0].n_train

    learning_rates = learning_rate * np.ones(n_task)
    learning_specs = []

    for i in range(n_task):
        opt = op.SGDOptimizer(learning_rates[i]).build()
        opt_spec = spec.OptimizerSpec(opt, d_in)
        learning_specs.append(spec.LearningSpec(n_epoch, n_batch, n_train, n_task, model_dir, opt_spec))

    meta_opt = op.SGDOptimizer(meta_learning_rate).build()
    meta_opt_spec = spec.OptimizerSpec(meta_opt, d_in)
    meta_learning_spec = spec.LearningSpec(n_epoch, n_batch, n_task, n_train, model_dir, meta_opt_spec)

    my_grouplearner = grouplearner.GroupHMTrainLearner(set_of_datasets, learning_specs, n_task, run_config,
                                                       meta_learning_spec)
    my_grouplearner.train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

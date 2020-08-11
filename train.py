import tensorflow as tf
import numpy as np
import argparse
import importlib
import sys
import logging

from optimizer import optimizer as op
from optimizer import spec
from optimizer import metric

from result import logger


def main(argv):
    parser = argparse.ArgumentParser(description='Homeostatic Synapse')

    # model parameters
    parser.add_argument('--model', type=str, default='Single', help='main learner')
    parser.add_argument('--alpha', type=float, default=1.0, help='Intensity of Regularization')

    # data parameters
    parser.add_argument('--data', type=str, default='RandMNISTPERM', help='Type of Dataset')

    # optimizer parameters
    parser.add_argument('--n_epoch', type=int, default=1, help='Number of epochs per task')
    parser.add_argument('--n_batch', type=int, default=10, help='batch size')
    parser.add_argument('--n_fed_batch', type=int, default=10, help='batch size for Fed learning')
    parser.add_argument('--lr', type=float, default=5e-2, help='SGD learning rate')

    # experiment parameters
    parser.add_argument('--n_task', type=int, default=10, help='Number of tasks')
    parser.add_argument('--n_block', type=int, default=7, help='Number of blocks in BPERM')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--save_path', type=str, default='results', help='save models')

    args = parser.parse_args()

    seed = args.seed
    alpha = args.alpha
    learning_rate = args.lr
    n_epoch = args.n_epoch
    n_batch = args.n_batch
    n_task = args.n_task
    n_block = args.n_block
    n_fed_batch = args.n_fed_batch
    save_path = args.save_path

    print("n_batch: ", n_batch)
    print("fed_batch: ", n_fed_batch)
    print("learning_rate: ", learning_rate)

    model_dir = args.model + args.data
    np.random.seed(seed)
    DataClass = getattr(importlib.import_module('dataset.set_of_dataset'), 'SetOf' + args.data)

    if args.data[-5:] == 'BPERM':
        set_of_datasets = DataClass(n_task, n_block)        # For Block-wise Permutation
    else:
        set_of_datasets = DataClass(n_task)

    n_train = set_of_datasets.list[0].n_train
    d_in = set_of_datasets.list[0].d_in

    learning_rates = learning_rate * np.ones(n_task)
    learning_specs = []

    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=int(6000))

    for i in range(n_task):
        opt = op.SGDOptimizer(learning_rates[i]).build()
        opt_spec = spec.OptimizerSpec(opt, d_in)
        learning_specs.append(spec.LearningSpec(n_epoch, n_batch, n_train, n_task,
                                                model_dir, opt_spec, n_fed_batch, alpha))

    ModelClass = getattr(importlib.import_module('model.grouplearner'), 'Group'+args.model+'Learner')
    my_grouplearner = ModelClass(set_of_datasets, learning_specs, n_task, run_config)

    accuracy_matrix = my_grouplearner.train_and_evaluate()

    avg_acc = metric.AverageAccuracy(accuracy_matrix).compute()
    tot_acc = metric.TotalAccuracy(accuracy_matrix).compute()
    avg_forget = metric.AverageForgetting(accuracy_matrix).compute()
    tot_forget = metric.TotalForgetting(accuracy_matrix).compute()

    metric_list = [avg_acc, tot_acc, avg_forget, tot_forget]
    filepath = save_path + "/" + model_dir + str(n_fed_batch) + "_" + str(seed) + ".txt"
    logger.save(filepath, model_dir, accuracy_matrix, metric_list, seed, learning_specs, 0, n_block)


if __name__ == '__main__':
    logging.getLogger("tensorflow").setLevel(logging.INFO)
    main(sys.argv[1])

import tensorflow as tf
import numpy as np
import argparse
import importlib
import sys
import logging
import result.logger as log

from optimizer import optimizer as op
from optimizer import spec


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
    parser.add_argument('--n_fed_step', type=int, default=600, help='step per each round for Fed learning')
    parser.add_argument('--n_fed_round', type=int, default=1, help='iteration round for Fed learning')
    parser.add_argument('--lr', type=float, default=5e-2, help='SGD learning rate')

    # experiment parameters
    parser.add_argument('--n_task', type=int, default=10, help='Number of tasks')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--save_path', type=str, default='result', help='save models')

    args = parser.parse_args()

    seed = args.seed
    alpha = args.alpha
    learning_rate = args.lr
    n_epoch = args.n_epoch
    n_batch = args.n_batch
    n_task = args.n_task
    n_fed_step = args.n_fed_step
    n_fed_round = args.n_fed_round
    save_path = args.save_path

    print(args)

    model_dir = args.model + args.data
    np.random.seed(seed)
    DataClass = getattr(importlib.import_module('dataset.set_of_dataset'), 'SetOf' + args.data)

    set_of_datasets = DataClass(n_task)

    n_train = set_of_datasets.list[0].n_train
    d_in = set_of_datasets.list[0].d_in

    learning_rates = learning_rate * np.ones(n_task)
    learning_specs = []

    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=int(n_train/n_batch))

    for i in range(n_task):
        opt = op.SGDOptimizer(learning_rates[i])
        opt_spec = spec.OptimizerSpec(opt, d_in)
        learning_specs.append(spec.LearningSpec(n_epoch, n_batch, n_train, n_task,
                                                model_dir, opt_spec, n_fed_step, n_fed_round, alpha))

    ModelClass = getattr(importlib.import_module('model.grouplearner'), 'Group'+args.model+'Learner')
    my_grouplearner = ModelClass(set_of_datasets, learning_specs, n_task, run_config)

    accuracy_vector = my_grouplearner.train_and_evaluate()
    print(accuracy_vector)
    filepath = args.model + ".txt"
    f = open(filepath, 'a')
    log.save_vector(accuracy_vector, len(accuracy_vector), f)


if __name__ == '__main__':
    logging.getLogger("tensorflow").setLevel(logging.INFO)
    main(sys.argv[1])

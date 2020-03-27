import tensorflow as tf
import numpy as np

from dataset import dataset
from model import learner
from optimizer import spec
from optimizer import optimizer as op

def main(argv):
    print(argv)
    seed = int(argv[1])
    learning_rate = 5e-2
    n_epoch = 1
    n_batch = 100
    n_task = 1

    np.random.seed(seed)
    model_dir = "single"

    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=100)

    # single_dataset = dataset.RandColPermMnist()

    # single_dataset = dataset.RandRowPermMnist()

    # single_dataset = dataset.RandPermMnist()

    n_grid = 7
    single_dataset = dataset.RandGridPermMnist(n_grid)

    d_in = single_dataset.d_in
    my_opt = op.SGDOptimizer().build(learning_rate)
    my_opt_spec = spec.OptimizerSpec(my_opt, d_in)
    my_learning_spec = spec.LearningSpec(n_epoch, n_batch, n_task, model_dir, my_opt_spec)

    base_learner = learner.SingleEstimatorLearner(single_dataset, my_learning_spec, run_config)

    base_learner.train()

    result = base_learner.evaluate()
    print(result)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

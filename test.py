import tensorflow as tf
import numpy as np

from dataset import dataset as ds
from model import learner
from optimizer import optimizer as op
from optimizer import spec


def main(argv):
    n_batch = 100
    model_dir = "test"
    np.random.seed(0)
    n_epoch = 1
    n_task = 1
    lr = 5e-2

    my_dataset = ds.RandRotaCIFAR10()
    n_train = my_dataset.n_train
    d_in = my_dataset.d_in

    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=int(n_train/n_batch))

    opt = op.SGDOptimizer(lr).build()
    opt_spec = spec.OptimizerSpec(opt, d_in)
    learning_spec =spec.LearningSpec(n_epoch, n_batch, n_task, model_dir, opt_spec)

    my_learner = learner.SingleEstimatorLearner(my_dataset, learning_spec, run_config)
    my_learner.train()
    result = my_learner.evaluate()
    print(result['accuracy'])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

import dataset
import tensorflow as tf
import learner
import optimizer as op
import numpy as np


def main(unused_argv):
    learning_rate = 5e-3
    n_epoch = 1
    n_task = 10
    n_batch = n_task * 10
    model_dir = "multi10"
    pre_model_dir = model_dir

    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=6000)

    set_single_dataset = dataset.SetOfRandRotaMnist(n_task)

    my_opt = op.SGDOptimizer().build(learning_rate)
    my_opt_spec = learner.OptimizerSpec(my_opt, learning_rate, pre_model_dir)
    my_learning_spec = learner.LearningSpec(n_epoch, n_batch, n_task, model_dir, my_opt_spec)

    accuracy_matrix = np.zeros(n_task, dtype=np.float32)

    multi_dataset = set_single_dataset.concat()
    multi_learner = learner.MultiEstimatorLearner(multi_dataset, my_learning_spec, run_config)
    multi_learner.train()

    for i in range(n_task):
        eval_learner = learner.SingleEstimatorLearner(set_single_dataset.list[i], my_learning_spec, run_config)
        result = eval_learner.evaluate()
        accuracy_matrix[i] = result['accuracy']

    np.set_printoptions(precision=4)
    print(accuracy_matrix)
    print(accuracy_matrix.sum() / 10.0)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

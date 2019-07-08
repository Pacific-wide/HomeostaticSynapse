import dataset
import tensorflow as tf
import learner
import optimizer as op
import numpy as np


def main(unused_argv):
    learning_rate = 1e-2
    n_epoch = 1
    n_batch = 10
    n_task = 10

    run_config = tf.estimator.RunConfig(model_dir="result", save_checkpoints_steps=6000)

    set_single_dataset = dataset.SetOfRandPermMnist(n_task)

    my_opt = op.SGDOptimizer().build(learning_rate)
    my_opt_spec = learner.OptimizerSpec(my_opt, learning_rate)
    my_learning_spec = learner.LearningSpec(n_epoch, n_batch, my_opt_spec)

    accuracy_matrix = np.zeros((n_task, n_task), dtype=np.float32)

    for i in range(n_task):
        single_dataset = set_single_dataset.list[i]
        single_learner = learner.SingleEstimatorLearner(single_dataset, my_learning_spec, run_config)
        single_learner.train()

        for j in range(i+1):
            eval_learner = learner.SingleEstimatorLearner(set_single_dataset.list[j], my_learning_spec, run_config)
            result = eval_learner.evaluate()
            accuracy_matrix[i, j] = result['accuracy']

    np.set_printoptions(precision=4)
    print(accuracy_matrix)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

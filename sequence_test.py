import dataset
import tensorflow as tf
import learner
import optimizer as op
import numpy as np


def main(unused_argv):
    base_learning_rate = 3e-3
    learning_rate = 3e-3
    n_epoch = 1
    n_batch = 10
    n_task = 5
    model_dir = "ewc"
    pre_model_dir = model_dir

    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=6000)

    set_single_dataset = dataset.SetOfRandRotaMnist(n_task)

    base_opt = op.SGDOptimizer().build(base_learning_rate)
    base_opt_spec = learner.OptimizerSpec(base_opt, learning_rate, pre_model_dir)
    base_learning_spec = learner.LearningSpec(n_epoch, n_batch, n_task, model_dir, base_opt_spec)

    my_opt = op.SGDOptimizer().build(learning_rate)
    my_opt_spec = learner.OptimizerSpec(my_opt, learning_rate, pre_model_dir)
    my_learning_spec = learner.LearningSpec(n_epoch, n_batch, n_task, model_dir, my_opt_spec)

    accuracy_matrix = np.zeros((n_task, n_task), dtype=np.float32)

    base_dataset = set_single_dataset.list[0]
    base_learner = learner.SingleEstimatorLearner(base_dataset, base_learning_spec, run_config)
    base_learner.train()

    eval_learner = learner.EWCEstimatorLearner(base_dataset, base_learning_spec, run_config)
    result = eval_learner.evaluate()
    accuracy_matrix[0, 0] = result['accuracy']

    for i in range(1, n_task):
        single_dataset = set_single_dataset.list[i]
        single_learner = learner.EWCEstimatorLearner(single_dataset, my_learning_spec, run_config)
        single_learner.train()

        for j in range(i+1):
            eval_learner = learner.SingleEstimatorLearner(set_single_dataset.list[j], my_learning_spec, run_config)
            result = eval_learner.evaluate()
            accuracy_matrix[i, j] = result['accuracy']

    np.set_printoptions(precision=4)
    print(accuracy_matrix)

    n_total = n_task*(n_task+1) / 2.0
    print(accuracy_matrix.sum()/n_total)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

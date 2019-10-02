import dataset
import tensorflow as tf
import learner
import optimizer as op
import numpy as np


def main(unused_argv):
    odd_learning_rate = 5e-2
    even_learning_rate = 5e-4
    n_epoch = 1
    n_batch = 10
    n_task = 10
    model_dir = "ewc"
    pre_model_dir = model_dir

    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=6000)

    set_single_dataset = dataset.SetOfAlternativeMnist(n_task)
    base_dataset = set_single_dataset.list[0]

    d_in = base_dataset.d_in

    odd_opt = op.SGDOptimizer().build(odd_learning_rate)
    odd_opt_spec = learner.OptimizerSpec(odd_opt, d_in, pre_model_dir)
    odd_learning_spec = learner.LearningSpec(n_epoch, n_batch, n_task, model_dir, odd_opt_spec)

    even_opt = op.SGDOptimizer().build(even_learning_rate)
    even_opt_spec = learner.OptimizerSpec(even_opt, d_in, pre_model_dir)
    even_learning_spec = learner.LearningSpec(n_epoch, n_batch, n_task, model_dir, even_opt_spec)

    accuracy_matrix = np.zeros((n_task, n_task), dtype=np.float32)

    base_learner = learner.EWCEstimatorLearner(base_dataset, odd_learning_spec, run_config)
    base_learner.train()

    eval_learner = learner.EWCEstimatorLearner(base_dataset, odd_learning_spec, run_config)
    result = eval_learner.evaluate()
    accuracy_matrix[0, 0] = result['accuracy']

    for i in range(1, n_task):
        single_dataset = set_single_dataset.list[i]

        if i % 2 == 0:
            single_learner = learner.EWCEstimatorLearner(single_dataset, odd_learning_spec, run_config)
        else:
            single_learner = learner.EWCEstimatorLearner(single_dataset, even_learning_spec, run_config)

        single_learner.train()

        for j in range(i+1):
            eval_learner = learner.SingleEstimatorLearner(set_single_dataset.list[j], odd_learning_spec, run_config)
            result = eval_learner.evaluate()
            accuracy_matrix[i, j] = result['accuracy']

    np.set_printoptions(precision=4)
    print(accuracy_matrix)
    n_total = n_task*(n_task+1) / 2.0
    print(accuracy_matrix.sum()/n_total)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

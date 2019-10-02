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

    # model path
    model_dir = "meta_test"
    pre_model_dir = "meta"

    # config
    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=6000)
    ws0 = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=pre_model_dir, vars_to_warm_start=["meta"])
    ws1 = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=pre_model_dir, vars_to_warm_start=["meta", "main"])

    # generate sequence dataset
    set_single_dataset = dataset.SetOfAlternativeMnist(n_task)
    base_dataset = set_single_dataset.list[0]
    d_in = base_dataset.d_in
    accuracy_matrix = np.zeros((n_task, n_task))

    # learning specs
    odd_opt = op.SGDOptimizer().build(odd_learning_rate)
    odd_opt_spec = learner.OptimizerSpec(odd_opt, d_in, pre_model_dir)
    odd_learning_spec = learner.LearningSpec(n_epoch, n_batch, n_task, model_dir, odd_opt_spec)

    even_opt = op.SGDOptimizer().build(even_learning_rate)
    even_opt_spec = learner.OptimizerSpec(even_opt, d_in, pre_model_dir)
    even_learning_spec = learner.LearningSpec(n_epoch, n_batch, n_task, model_dir, even_opt_spec)

    # base test
    base_test_learner = learner.MetaWarmTestEstimatorLearner(base_dataset, odd_learning_spec, run_config, ws0)
    base_test_learner.train()

    eval_learner = learner.SingleEstimatorLearner(base_dataset, odd_learning_spec, run_config)
    result = eval_learner.evaluate()
    accuracy_matrix[0, 0] = result['accuracy']

    for i in range(1, n_task):
        single_dataset = set_single_dataset.list[i]

        if i % 2 == 0:
            meta_test_learner = learner.MetaWarmTestEstimatorLearner(single_dataset, odd_learning_spec, run_config, ws1)
        else:
            meta_test_learner = learner.MetaWarmTestEstimatorLearner(single_dataset, even_learning_spec, run_config, ws1)

        meta_test_learner.train()

        for j in range(i+1):
            eval_learner = learner.SingleEstimatorLearner(set_single_dataset.list[j], odd_learning_spec, run_config)
            result = eval_learner.evaluate()
            accuracy_matrix[i, j] = result['accuracy']

    np.set_printoptions(precision=4)
    print(accuracy_matrix)
    n_total = n_task * (n_task + 1) / 2.0
    print(accuracy_matrix.sum() / n_total)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

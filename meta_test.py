import dataset
import tensorflow as tf
import learner
import optimizer as op
import numpy as np


def main(unused_argv):
    learning_rate = 5e-2
    n_epoch = 1
    n_batch = 10
    n_task = 10

    # model path
    model_dir = "meta_rota"
    pre_model_dir = model_dir

    # config
    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=6000)
    ws0 = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=model_dir, vars_to_warm_start=["meta"])
    ws1 = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=model_dir, vars_to_warm_start=['main', "meta"])

    # learning specs
    opt = op.SGDOptimizer().build(learning_rate)
    opt_spec = learner.OptimizerSpec(opt, learning_rate, pre_model_dir)
    learning_spec = learner.LearningSpec(n_epoch, n_batch, n_task, model_dir, opt_spec)

    # generate sequence dataset
    set_single_dataset = dataset.SetOfRandPermMnist(n_task)
    accuracy_matrix = np.zeros((n_task, n_task))

    base_dataset = set_single_dataset.list[0]
    base_test_learner = learner.MetaTestEstimatorLearner(base_dataset, learning_spec, run_config, ws1)
    base_test_learner.train()

    eval_learner = learner.SingleEstimatorLearner(base_dataset, learning_spec, run_config)
    accuracy_matrix[0, 0] = eval_learner.evaluate()['accuracy']

    for i in range(1, n_task):
        single_dataset = set_single_dataset.list[i]
        meta_test_learner = learner.MetaTestEstimatorLearner(single_dataset, learning_spec, run_config, ws1)
        meta_test_learner.train()

        for j in range(i+1):
            eval_learner = learner.SingleEstimatorLearner(set_single_dataset.list[j], learning_spec, run_config)
            result = eval_learner.evaluate()
            accuracy_matrix[i, j] = result['accuracy']

    np.set_printoptions(precision=4)
    print(accuracy_matrix)
    print(accuracy_matrix.sum() / 55.0)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

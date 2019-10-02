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
    model_dir = "meta_single"
    pre_model_dir = "meta"

    # config
    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=6000)
    ws0 = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=pre_model_dir, vars_to_warm_start=["meta"])

    # generate sequence dataset
    single_dataset = dataset.RandPermMnist()
    d_in = single_dataset.d_in

    # learning specs
    opt = op.SGDOptimizer().build(learning_rate)
    opt_spec = learner.OptimizerSpec(opt, d_in, model_dir)
    learning_spec = learner.LearningSpec(n_epoch, n_batch, n_task, model_dir, opt_spec)

    # base test
    test_learner = learner.MetaTestEstimatorLearner(single_dataset, learning_spec, run_config, ws0)
    test_learner.train()

    eval_learner = learner.SingleEstimatorLearner(single_dataset, learning_spec, run_config)
    accuracy = eval_learner.evaluate()['accuracy']
    print(accuracy)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

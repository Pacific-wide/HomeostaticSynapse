import dataset
import tensorflow as tf
import learner
import optimizer as op


def main(unused_argv):
    learning_rate = 1e-1
    n_epoch = 1
    n_batch = 10
    n_task = 1
    model_dir = "single"
    pre_model_dir = model_dir

    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=6000)

    # single_dataset = dataset.RandPermMnist()

    single_dataset = dataset.SVHN()

    d_in = single_dataset.d_in
    my_opt = op.SGDOptimizer().build(learning_rate)
    my_opt_spec = learner.OptimizerSpec(my_opt, learning_rate, d_in, pre_model_dir)
    my_learning_spec = learner.LearningSpec(n_epoch, n_batch, n_task, model_dir, my_opt_spec)

    single_learner = learner.SingleEstimatorLearner(single_dataset, my_learning_spec, run_config)

    single_learner.train()

    result = single_learner.evaluate()
    print(result)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

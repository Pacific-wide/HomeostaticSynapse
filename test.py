import dataset
import tensorflow as tf
import learner
import optimizer as op


def main(unused_argv):
    learning_rate = 1e-2
    n_epoch = 1
    n_batch = 10

    run_config = tf.estimator.RunConfig(model_dir="log", save_checkpoints_steps=6000)

    single_dataset = dataset.RandPermMnist()

    my_opt = op.SGDOptimizer().build(learning_rate)
    my_opt_spec = learner.OptimizerSpec(my_opt, learning_rate)
    my_learning_spec = learner.LearningSpec(n_epoch, n_batch, my_opt_spec)

    single_learner = learner.EstimatorLearner(single_dataset, my_learning_spec, run_config)

    single_learner.train()

    result = single_learner.evaluate()
    print(result)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

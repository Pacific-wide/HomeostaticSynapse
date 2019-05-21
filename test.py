import dataset
import tensorflow as tf
import learner
import net
import optimizer as op


def main(unused_argv):
    learning_rate = 1e-3
    n_epoch = 1
    n_batch = 10

    single_dataset = dataset.RandPermMnist()
    my_model = net.FCN()

    my_opt = op.SGDOptimizer().build(learning_rate)
    my_opt_spec = learner.OptimizerSpec(my_opt, learning_rate)
    my_learning_spec = learner.LearningSpec(n_epoch, n_batch, my_opt_spec)

    single_learner = learner.EstimatorLearner(my_model, single_dataset, my_learning_spec)
    single_learner.train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

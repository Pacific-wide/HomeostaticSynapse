from absl import flags
import dataset
import tensorflow as tf

flags.DEFINE_integer('n_task', '10', 'Number of tasks in evaluation sequence.')
FLAGS = flags.FLAGS

def main(unused_argv):
    single_dataset = dataset.RandPermMnist()
    my_model = model.Model()
    single_learner = EstimatorLearner(my_model, single_dataset)
    single_learner.train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

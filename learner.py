import abc
import tensorflow as tf
import model_fn


class NNLearner(object):
    def __init__(self, dataset, learning_spec):
        self.dataset = dataset
        self.learning_spec = learning_spec

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass


class EstimatorLearner(NNLearner):
    def __init__(self, dataset, learning_spec, run_config):
        super(EstimatorLearner, self).__init__(dataset, learning_spec)
        self.estimator = tf.estimator.Estimator(model_fn=self.model_fn, config=run_config)

    def train(self):
        self.estimator.train(input_fn=self.train_input_fn)

    def evaluate(self):
        return self.estimator.evaluate(input_fn=self.eval_input_fn, steps=1000)

    def train_input_fn(self):
        tf_train = tf.data.Dataset.from_tensor_slices((self.dataset.x_train, self.dataset.y_train))
        tf_train = tf_train.shuffle(600000).repeat(self.learning_spec.n_epoch).batch(self.learning_spec.n_batch)

        return tf_train

    def eval_input_fn(self):
        tf_eval = tf.data.Dataset.from_tensor_slices((self.dataset.x_test, self.dataset.y_test))
        tf_eval = tf_eval.batch(self.learning_spec.n_batch)

        return tf_eval

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.SingleModelFNCreator(features, labels, mode, self.learning_spec.optimizer_spec)
        return model_fn_creator.create()


class OptimizerSpec(object):
    def __init__(self, optimizer, learning_rate):
        self.optimizer = optimizer
        self.learning_rate = learning_rate


class LearningSpec(object):
    def __init__(self, n_epoch, n_batch, optimizer_spec):
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.optimizer_spec = optimizer_spec


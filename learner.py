import abc
import tensorflow as tf
import model_fn


class NNLearner(object):
    def __init__(self, dataset, model, learning_spec):
        self.dataset = dataset
        self.model = model
        self.learning_spec = learning_spec

    @abc.abstractmethod
    def train(self):
        pass


class EstimatorLearner(NNLearner):
    def __init__(self, dataset, model, learning_spec):
        super(EstimatorLearner, self).__init__(dataset, model, learning_spec)
        self.estimator = tf.estimator.Estimator(model_fn=self.model_fn)

    def train(self):
        self.estimator.train(input_fn=self.input_fn)

    def input_fn(self):
        self.dataset = tf.data.Dataset.from_tensor_slices(self.dataset).batch(self.learning_spec.n_batch)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.SingleModelFNCreator(self.model, features, labels, mode,
                                                         self.learning_spec.OptimizerSpec)
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

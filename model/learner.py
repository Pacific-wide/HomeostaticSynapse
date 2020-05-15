import abc
import tensorflow as tf
from model import model_fn


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
        tf_train = tf_train.repeat(self.learning_spec.n_epoch).batch(self.learning_spec.n_batch)

        return tf_train

    def eval_input_fn(self):
        tf_eval = tf.data.Dataset.from_tensor_slices((self.dataset.x_test, self.dataset.y_test))
        tf_eval = tf_eval.batch(self.learning_spec.n_batch)

        return tf_eval

    def model_fn(self, features, labels, mode):
        pass


class WarmStartEstimatorLearner(EstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config, ws):
        super(WarmStartEstimatorLearner, self).__init__(dataset, learning_spec, run_config)
        self.estimator = tf.estimator.Estimator(model_fn=self.model_fn, config=run_config, warm_start_from=ws)


class SingleEstimatorLearner(EstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config):
        super(SingleEstimatorLearner, self).__init__(dataset, learning_spec, run_config)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.SingleModelFNCreator(features, labels, mode, self.learning_spec)

        return model_fn_creator.create()


class EWCEstimatorLearner(EstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config):
        super(EWCEstimatorLearner, self).__init__(dataset, learning_spec, run_config)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.EWCModelFNCreator(features, labels, mode, self.learning_spec)

        return model_fn_creator.create()


class FullEWCEstimatorLearner(EWCEstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config, i_task):
        super(FullEWCEstimatorLearner, self).__init__(dataset, learning_spec, run_config)
        self.i_task = i_task

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.FullEWCModelFNCreator(features, labels, mode, self.learning_spec, self.i_task)

        return model_fn_creator.create()


class MultiEstimatorLearner(EstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config):
        super(MultiEstimatorLearner, self).__init__(dataset, learning_spec, run_config)
        self.x_max = dataset.x_train.shape[0]

    def train_input_fn(self):
        tf_train = tf.data.Dataset.from_tensor_slices((self.dataset.x_train, self.dataset.y_train))
        tf_train = tf_train.shuffle(self.x_max*self.learning_spec.n_task).repeat(self.learning_spec.n_epoch).batch(self.learning_spec.n_batch)

        return tf_train

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.SingleModelFNCreator(features, labels, mode, self.learning_spec)

        return model_fn_creator.create()


class JointEstimatorLearner(MultiEstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config):
        super(MultiEstimatorLearner, self).__init__(dataset, learning_spec, run_config)
        self.x_max = dataset.x_train.shape[0]

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.JointModelFNCreator(features, labels, mode, self.learning_spec)

        return model_fn_creator.create()


class BaseEstimatorLearner(EstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config):
        super(BaseEstimatorLearner, self).__init__(dataset, learning_spec, run_config)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.BaseModelFNCreator(features, labels, mode, self.learning_spec)

        return model_fn_creator.create()


class FullBaseEstimatorLearner(BaseEstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config, i_task):
        super(BaseEstimatorLearner, self).__init__(dataset, learning_spec, run_config)
        self.i_task = i_task

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.FullBaseModelFNCreator(features, labels, mode, self.learning_spec, self.i_task)

        return model_fn_creator.create()


class MetaAlphaBaseEstimatorLearner(EstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config):
        super(MetaAlphaBaseEstimatorLearner, self).__init__(dataset, learning_spec, run_config)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.MetaAlphaModelFNCreator(features, labels, mode, self.learning_spec)

        return model_fn_creator.create()


class MetaAlphaWarmBaseEstimatorLearner(WarmStartEstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config, ws):
        super(MetaAlphaWarmBaseEstimatorLearner, self).__init__(dataset, learning_spec, run_config, ws)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.MetaAlphaModelFNCreator(features, labels, mode, self.learning_spec)

        return model_fn_creator.create()


class MetaAlphaWarmTestEstimatorLearner(WarmStartEstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config, ws):
        super(MetaAlphaWarmTestEstimatorLearner, self).__init__(dataset, learning_spec, run_config, ws)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.MetaAlphaTestModelFNCreator(features, labels, mode,
                                                                self.learning_spec)

        return model_fn_creator.create()


class MetaAlphaTrainEstimatorLearner(EstimatorLearner):
    def __init__(self, datasets, learning_spec, meta_learning_spec, run_config):
        super(MetaAlphaTrainEstimatorLearner, self).__init__(datasets, learning_spec)
        self.learning_spec = learning_spec
        self.meta_learning_spec = meta_learning_spec
        self.estimator = tf.estimator.Estimator(model_fn=self.model_fn, config=run_config)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.MetaAlphaTrainModelFNCreator(features, labels, mode,
                                                                 self.learning_spec, self.meta_learning_spec)

        return model_fn_creator.create()

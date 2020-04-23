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
    def __init__(self, dataset, learning_spec, run_config):
        super(FullEWCEstimatorLearner, self).__init__(dataset, learning_spec, run_config)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.FullEWCModelFNCreator(features, labels, mode, self.learning_spec)

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


class BaseEstimatorLearner(EstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config):
        super(BaseEstimatorLearner, self).__init__(dataset, learning_spec, run_config)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.BaseModelFNCreator(features, labels, mode, self.learning_spec)

        return model_fn_creator.create()


class FullBaseEstimatorLearner(BaseEstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config):
        super(BaseEstimatorLearner, self).__init__(dataset, learning_spec, run_config)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.FullBaseModelFNCreator(features, labels, mode, self.learning_spec)

        return model_fn_creator.create()


class MetaGradientBaseEstimatorLearner(EstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config):
        super(MetaGradientBaseEstimatorLearner, self).__init__(dataset, learning_spec, run_config)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.MetaGradientModelFNCreator(features, labels, mode, self.learning_spec)

        return model_fn_creator.create()


class MetaGradientWarmBaseEstimatorLearner(WarmStartEstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config, ws):
        super(MetaGradientWarmBaseEstimatorLearner, self).__init__(dataset, learning_spec, run_config, ws)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.MetaGradientModelFNCreator(features, labels, mode, self.learning_spec)

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


class MetaGradientWarmTestEstimatorLearner(WarmStartEstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config, ws):
        super(MetaGradientWarmTestEstimatorLearner, self).__init__(dataset, learning_spec, run_config, ws)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.MetaGradientTestModelFNCreator(features, labels, mode,
                                                           self.learning_spec)

        return model_fn_creator.create()


class MetaGradientTestEstimatorLearner(EstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config):
        super(MetaGradientTestEstimatorLearner, self).__init__(dataset, learning_spec, run_config)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.MetaGradientTestModelFNCreator(features, labels, mode, self.learning_spec)

        return model_fn_creator.create()


class MetaAlphaWarmTestEstimatorLearner(WarmStartEstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config, ws):
        super(MetaAlphaWarmTestEstimatorLearner, self).__init__(dataset, learning_spec, run_config, ws)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.MetaAlphaTestModelFNCreator(features, labels, mode,
                                                                self.learning_spec)

        return model_fn_creator.create()


class MetaLearner(object):
    def __init__(self, datasets, meta_learning_spec):
        self.datasets = datasets
        self.meta_learning_spec = meta_learning_spec

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass


class MetaGradientTrainEstimatorLearner(MetaLearner):
    def __init__(self, datasets, learning_spec, meta_learning_spec, run_config):
        super(MetaGradientTrainEstimatorLearner, self).__init__(datasets, learning_spec)
        self.learning_spec = learning_spec
        self.meta_learning_spec = meta_learning_spec
        self.estimator = tf.estimator.Estimator(model_fn=self.model_fn, config=run_config)

    def train(self):
        self.estimator.train(input_fn=self.train_input_fn)

    def evaluate(self):
        return self.estimator.evaluate(input_fn=self.eval_input_fn, steps=1000)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.MetaGradientTrainModelFNCreator(features, labels, mode, self.learning_spec, self.meta_learning_spec)

        return model_fn_creator.create()

    def train_input_fn(self):
        tf_train0 = tf.data.Dataset.from_tensor_slices((self.datasets[0].x_train, self.datasets[0].y_train))
        tf_train1 = tf.data.Dataset.from_tensor_slices((self.datasets[1].x_train, self.datasets[1].y_train))

        dataset_tuple = (tf_train0, tf_train1)
        tf_comb_train = tf.data.Dataset.zip(dataset_tuple)
        tf_flat_train = tf_comb_train.flat_map(self.map_fn)
        total_tuple = (tf_flat_train, ) + dataset_tuple

        tf_train = tf.data.Dataset.zip(total_tuple)
        tf_train = tf_train.map(self.unfold_tuple)
        tf_train = tf_train.repeat(self.learning_spec.n_epoch).batch(2*self.learning_spec.n_batch)

        return tf_train

    def unfold_tuple(self, *x):
        t1, t2, t3 = x

        f1, l1 = t1
        f2, l2 = t2
        f3, l3 = t3

        return (f1, f2, f3), (l1, l2, l3)

    def map_fn(self, *z):
        x, y = zip(*z)

        return tf.data.Dataset.from_tensor_slices((tf.stack(x), tf.stack(y)))


class MetaAlphaTrainEstimatorLearner(MetaGradientTrainEstimatorLearner):
    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.MetaAlphaTrainModelFNCreator(features, labels, mode,
                                                                 self.learning_spec)

        return model_fn_creator.create()

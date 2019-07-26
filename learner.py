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
        tf_train = tf_train.repeat(self.learning_spec.n_epoch).batch(self.learning_spec.n_batch)

        return tf_train

    def eval_input_fn(self):
        tf_eval = tf.data.Dataset.from_tensor_slices((self.dataset.x_test, self.dataset.y_test))
        tf_eval = tf_eval.batch(self.learning_spec.n_batch)

        return tf_eval

    def model_fn(self, features, labels, mode):
        pass


class SingleEstimatorLearner(EstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config):
        super(SingleEstimatorLearner, self).__init__(dataset, learning_spec, run_config)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.SingleModelFNCreator(features, labels, mode, self.learning_spec.optimizer_spec)

        return model_fn_creator.create()


class EWCEstimatorLearner(EstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config):
        super(EWCEstimatorLearner, self).__init__(dataset, learning_spec, run_config)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.EWCModelFNCreator(features, labels, mode, self.learning_spec.optimizer_spec)

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
        model_fn_creator = model_fn.SingleModelFNCreator(features, labels, mode, self.learning_spec.optimizer_spec)

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


class MetaEstimatorLearner(MetaLearner):
    def __init__(self, datasets, learning_spec, meta_learning_spec, run_config):
        super(MetaEstimatorLearner, self).__init__(datasets, learning_spec)
        self.learning_spec = learning_spec
        self.meta_learning_spec = meta_learning_spec
        self.estimator = tf.estimator.Estimator(model_fn=self.model_fn, config=run_config)

    def train(self):
        self.estimator.train(input_fn=self.train_input_fn)

    def evaluate(self):
        return self.estimator.evaluate(input_fn=self.eval_input_fn, steps=1000)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.MetaModelFNCreator(features, labels, mode, self.learning_spec.optimizer_spec,
                                                       self.meta_learning_spec.optimizer_spec)

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


class OptimizerSpec(object):
    def __init__(self, optimizer, learning_rate, pre_model_dir):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.pre_model_dir = pre_model_dir


class LearningSpec(object):
    def __init__(self, n_epoch, n_batch, n_task, model_dir, optimizer_spec):
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.n_task = n_task
        self.optimizer_spec = optimizer_spec
        self.model_dir = model_dir


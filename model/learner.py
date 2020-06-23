import abc
import tensorflow as tf
import numpy as np
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
        tf_eval = tf_eval.batch(10)

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


class OEWCEstimatorLearner(EstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config):
        super(OEWCEstimatorLearner, self).__init__(dataset, learning_spec, run_config)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.OEWCModelFNCreator(features, labels, mode, self.learning_spec)

        return model_fn_creator.create()


class CenterEWCEstimatorLearner(OEWCEstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config, i_task):
        super(CenterEWCEstimatorLearner, self).__init__(dataset, learning_spec, run_config)
        self.i_task = i_task

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.CenterEWCModelFNCreator(features, labels, mode, self.learning_spec, self.i_task)

        return model_fn_creator.create()


class EWCEstimatorLearner(OEWCEstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config, i_task):
        super(EWCEstimatorLearner, self).__init__(dataset, learning_spec, run_config)
        self.i_task = i_task

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.EWCModelFNCreator(features, labels, mode, self.learning_spec, self.i_task)

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


class FedEstimatorLearner(EstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config):
        super(FedEstimatorLearner, self).__init__(dataset, learning_spec, run_config)
        self.tf_train_list = []

    def train_input_fn(self):
        tf_train = self.combine_dataset(self.dataset)
        tf_train = tf_train.repeat(self.learning_spec.n_epoch).batch(self.learning_spec.n_batch)

        return tf_train

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.SingleModelFNCreator(features, labels, mode, self.learning_spec)

        return model_fn_creator.create()

    def combine_dataset(self, dataset):
        x_batchs = []
        y_batchs = []
        n_batch = self.learning_spec.n_batch

        for data in dataset:
            for i in range(int(self.learning_spec.n_train/n_batch)):
                x_batchs.append(data.x_train[n_batch*i:n_batch*(i+1)])
                y_batchs.append(data.y_train[n_batch*i:n_batch*(i+1)])

        np_x_train = np.concatenate(x_batchs, axis=0)
        np_y_train = np.concatenate(y_batchs, axis=0)

        print(np_x_train.shape)
        print(np_y_train.shape)

        return tf.data.Dataset.from_tensor_slices((np_x_train, np_y_train))


class IMMEstimatorLearner(EstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config, i_task):
        super(IMMEstimatorLearner, self).__init__(dataset, learning_spec, run_config)
        self.i_task = i_task

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.IMMModelFNCreator(features, labels, mode, self.learning_spec, self.i_task)

        return model_fn_creator.create()


class BaseEstimatorLearner(EstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config):
        super(BaseEstimatorLearner, self).__init__(dataset, learning_spec, run_config)

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.BaseModelFNCreator(features, labels, mode, self.learning_spec)

        return model_fn_creator.create()


class CenterBaseEstimatorLearner(EstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config, i_task):
        super(CenterBaseEstimatorLearner, self).__init__(dataset, learning_spec, run_config)
        self.i_task = i_task

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.CenterBaseModelFNCreator(features, labels, mode, self.learning_spec, self.i_task)

        return model_fn_creator.create()


class FullBaseEstimatorLearner(BaseEstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config, i_task):
        super(BaseEstimatorLearner, self).__init__(dataset, learning_spec, run_config)
        self.i_task = i_task

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.FullBaseModelFNCreator(features, labels, mode, self.learning_spec, self.i_task)

        return model_fn_creator.create()


class MetaAlphaBaseEstimatorLearner(EstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config, i_task):
        super(MetaAlphaBaseEstimatorLearner, self).__init__(dataset, learning_spec, run_config)
        self.i_task = i_task

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.MetaAlphaModelFNCreator(features, labels, mode, self.learning_spec, self.i_task)

        return model_fn_creator.create()


class MetaAlphaWarmBaseEstimatorLearner(WarmStartEstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config, ws, i_task):
        super(MetaAlphaWarmBaseEstimatorLearner, self).__init__(dataset, learning_spec, run_config, ws)
        self.i_task = i_task

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.MetaAlphaModelFNCreator(features, labels, mode, self.learning_spec, self.i_task)

        return model_fn_creator.create()


class MetaAlphaWarmTestEstimatorLearner(WarmStartEstimatorLearner):
    def __init__(self, dataset, learning_spec, run_config, ws, i_task):
        super(MetaAlphaWarmTestEstimatorLearner, self).__init__(dataset, learning_spec, run_config, ws)
        self.i_task = i_task

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.MetaAlphaTestModelFNCreator(features, labels, mode,
                                                                self.learning_spec, self.i_task)

        return model_fn_creator.create()


class MetaAlphaTrainEstimatorLearner(EstimatorLearner):
    def __init__(self, dataset, learning_spec, meta_learning_spec, run_config, i_task):
        super(MetaAlphaTrainEstimatorLearner, self).__init__(dataset, learning_spec, run_config)
        self.meta_learning_spec = meta_learning_spec
        self.i_task = i_task

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.MetaAlphaTrainModelFNCreator(features, labels, mode,
                                                                 self.learning_spec,
                                                                 self.meta_learning_spec,
                                                                 self.i_task)

        return model_fn_creator.create()

    def train_input_fn(self):
        tf_train0 = tf.data.Dataset.from_tensor_slices((self.dataset[0].x_train, self.dataset[0].y_train))
        tf_train1 = tf.data.Dataset.from_tensor_slices((self.dataset[1].x_train, self.dataset[1].y_train))

        dataset_tuple = (tf_train0, tf_train1)
        tf_comb_train = tf.data.Dataset.zip(dataset_tuple)
        tf_flat_train = tf_comb_train.flat_map(self.map_fn)
        total_tuple = (tf_flat_train,) + dataset_tuple

        tf_train = tf.data.Dataset.zip(total_tuple)
        tf_train = tf_train.map(self.unfold_tuple)
        tf_train = tf_train.repeat(self.learning_spec.n_epoch).batch(2 * self.learning_spec.n_batch)

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

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

    def train_input_fn(self):
        tf_train = tf.data.Dataset.from_tensor_slices((self.dataset.x_train, self.dataset.y_train))
        tf_train = tf_train.shuffle(60000*self.learning_spec.n_task).repeat(self.learning_spec.n_epoch).batch(self.learning_spec.n_batch)

        return tf_train

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.SingleModelFNCreator(features, labels, mode, self.learning_spec.optimizer_spec)

        return model_fn_creator.create()


class MetaEstimatorLearner(object):
    def __init__(self, learner1, learner2):
        self.learner1 = learner1
        self.learner2 = learner2
        self.n_epoch = learner1.learning_spec.n_epoch
        self.n_batch = learner1.learning_spec.n_batch
        self.n_task = learner1.learning_spec.n_task

    def meta_train_input_fn(self):
        tf_train1 = tf.data.Dataset.from_tensor_slices((self.learner1.dataset.x_train, self.learner1.dataset.y_train))
        tf_train2 = tf.data.Dataset.from_tensor_slices((self.learner2.dataset.x_train, self.learner2.dataset.y_train))

        task_tuple = (tf_train1, tf_train2)
        comb_train = tf.data.Dataset.zip(task_tuple)
        flat_train = comb_train.flat_map(self.map_fn)

        data_tuple = (flat_train,) + task_tuple
        final_train = tf.data.Dataset.zip(data_tuple)
        final_train = final_train.map(self.unfold_tuple)
        final_train = final_train.repeat(self.n_epoch).batch(2 * self.n_task * self.n_batch)

        return final_train

    def model_fn(self, features, labels, mode):
        model_fn_creator = model_fn.MetaModelFNCreator(features, labels, mode, self.learning_spec.optimizer_spec)

        return model_fn_creator.create()

    def map_fn(*z):
        x, y = zip(*z)
        return tf.data.Dataset.from_tensor_slices((tf.stack(x), tf.stack(y)))

    def unfold_tuple(*x):
        t1, t2, t3 = x
        f1, l1 = t1
        f2, l2 = t2
        f3, l3 = t3
        return (f1, f2, f3), (l1, l2, l3)


class OptimizerSpec(object):
    def __init__(self, optimizer, learning_rate):
        self.optimizer = optimizer
        self.learning_rate = learning_rate


class LearningSpec(object):
    def __init__(self, n_epoch, n_batch, n_task, optimizer_spec):
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.n_task = n_task
        self.optimizer_spec = optimizer_spec


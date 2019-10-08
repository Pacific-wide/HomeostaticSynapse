import learner
import numpy as np


class GroupLearner(object):
    def __init__(self, set_of_dataset, learning_spec, n_task, run_config):
        self.set_of_dataset = set_of_dataset
        self.learning_spec = learning_spec
        self.n_task = n_task
        self.run_config = run_config
        self.eval_matrix = np.zeros((self.n_task, self.n_task), dtype=np.float32)

    def train_and_evaluate(self):
        pass


class GroupSingleLearner(GroupLearner):
    def __init__(self, set_of_dataset, learning_spec, n_task, run_config):
        super(GroupSingleLearner, self).__init__(set_of_dataset, learning_spec, n_task, run_config)

    def train_and_evaluate(self):
        for i in range(self.n_task):
            dataset = self.set_of_dataset.list[i]
            single_learner = learner.SingleEstimatorLearner(dataset, self.learning_spec, self.run_config)
            single_learner.train()

            for j in range(i + 1):
                eval_learner = learner.SingleEstimatorLearner(self.set_of_dataset.list[j], self.learning_spec, self.run_config)
                result = eval_learner.evaluate()
                self.eval_matrix[i, j] = result['accuracy']

        return self.eval_matrix


class GroupEWCLearner(GroupLearner):
    def __init__(self, set_of_dataset, learning_spec, n_task, run_config):
        super(GroupEWCLearner, self).__init__(set_of_dataset, learning_spec, n_task, run_config)

    def base_train(self):
        base_dataset = self.set_of_dataset.list[0]
        base_learner = learner.BaseEstimatorLearner(base_dataset, self.learning_spec, self.run_config)
        base_learner.train()

        result = base_learner.evaluate()
        self.eval_matrix[0, 0] = result['accuracy']

    def train_and_evaluate(self):
        self.base_train()

        for i in range(1, self.n_task):
            dataset = self.set_of_dataset.list[i]
            single_learner = learner.EWCEstimatorLearner(dataset, self.learning_spec, self.run_config)
            single_learner.train()

            for j in range(i + 1):
                eval_learner = learner.SingleEstimatorLearner(self.set_of_dataset.list[j], self.learning_spec, self.run_config)
                result = eval_learner.evaluate()
                self.eval_matrix[i, j] = result['accuracy']

        return self.eval_matrix


class GroupMultiLearner(GroupLearner):
    def __init__(self, set_of_dataset, learning_spec, n_task, run_config):
        super(GroupMultiLearner, self).__init__(set_of_dataset, learning_spec, n_task, run_config)
        self.accuracy_vector = np.zeros(n_task, dtype=np.float32)

    def train_and_evaluate(self):
        multi_learner = learner.MultiEstimatorLearner(self.set_of_dataset.concat(), self.learning_spec, self.run_config)
        multi_learner.train()

        for i in range(self.n_task):
            eval_learner = learner.SingleEstimatorLearner(self.set_of_dataset.list[i], self.learning_spec, self.run_config)
            result = eval_learner.evaluate()
            self.accuracy_vector[i] = result['accuracy']

        return self.accuracy_vector


class GroupMetaTrainLearner(GroupLearner):
    def __init__(self, set_of_dataset, learning_spec, n_task, run_config, meta_learning_spec):
        super(GroupMetaTrainLearner, self).__init__(set_of_dataset, learning_spec, n_task, run_config)
        self.meta_learning_spec = meta_learning_spec

    def base_train(self):
        base_dataset = self.set_of_dataset.list[0]
        base_learner = learner.MetaBaseEstimatorLearner(base_dataset, self.learning_spec, self.run_config)
        base_learner.train()

    def train_and_evaluate(self):
        self.base_train()

        for i in range(0, self.n_task):
            dataset = self.set_of_dataset.list[i:i+2]
            meta_learner = learner.MetaTrainEstimatorLearner(dataset, self.learning_spec, self.meta_learning_spec, self.run_config)
            meta_learner.train()


class GroupMetaTestLearner(GroupLearner):
    def __init__(self, set_of_dataset, learning_spec, n_task, run_config, ws0, ws1):
        super(GroupMetaTestLearner, self).__init__(set_of_dataset, learning_spec, n_task, run_config)
        self.ws0 = ws0
        self.ws1 = ws1

    def base_train(self):
        base_dataset = self.set_of_dataset.list[0]
        base_learner = learner.MetaWarmBaseEstimatorLearner(base_dataset, self.learning_spec, self.run_config, self.ws0)
        base_learner.train()

        result = base_learner.evaluate()
        self.eval_matrix[0, 0] = result['accuracy']

    def train_and_evaluate(self):
        self.base_train()

        for i in range(1, self.n_task):
            dataset = self.set_of_dataset.list[i]
            meta_learner = learner.MetaWarmTestEstimatorLearner(dataset, self.learning_spec, self.run_config, self.ws1)
            meta_learner.train()

            for j in range(i + 1):
                eval_learner = learner.SingleEstimatorLearner(self.set_of_dataset.list[j], self.learning_spec, self.run_config)
                result = eval_learner.evaluate()
                self.eval_matrix[i, j] = result['accuracy']

        return self.eval_matrix

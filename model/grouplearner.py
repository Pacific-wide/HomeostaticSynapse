from model import learner
import numpy as np


class GroupLearner(object):
    def __init__(self, set_of_dataset, learning_specs, n_task, run_config):
        self.set_of_dataset = set_of_dataset
        self.learning_specs = learning_specs
        self.n_task = n_task
        self.run_config = run_config
        self.eval_matrix = np.zeros((self.n_task, self.n_task), dtype=np.float32)

    def train_and_evaluate(self):
        pass

    def evaluate(self, i):
        for j in range(i + 1):
            self.learning_specs[j].n_batch = 10
            eval_learner = learner.SingleEstimatorLearner(self.set_of_dataset.list[j], self.learning_specs[j],
                                                          self.run_config)
            result = eval_learner.evaluate()
            self.eval_matrix[i, j] = result['accuracy']


class GroupSingleLearner(GroupLearner):
    def __init__(self, set_of_dataset, learning_specs, n_task, run_config):
        super(GroupSingleLearner, self).__init__(set_of_dataset, learning_specs, n_task, run_config)

    def train_and_evaluate(self):
        for i in range(self.n_task):
            dataset = self.set_of_dataset.list[i]
            single_learner = learner.SingleEstimatorLearner(dataset, self.learning_specs[i], self.run_config)
            single_learner.train()

            self.evaluate(i)

        return self.eval_matrix


class GroupOEWCLearner(GroupLearner):
    def __init__(self, set_of_dataset, learning_specs, n_task, run_config):
        super(GroupOEWCLearner, self).__init__(set_of_dataset, learning_specs, n_task, run_config)

    def base_train(self):
        base_dataset = self.set_of_dataset.list[0]
        base_learner = learner.BaseEstimatorLearner(base_dataset, self.learning_specs[0], self.run_config)
        base_learner.train()

        result = base_learner.evaluate()
        self.eval_matrix[0, 0] = result['accuracy']

    def train_and_evaluate(self):
        self.base_train()

        for i in range(1, self.n_task):
            dataset = self.set_of_dataset.list[i]
            single_learner = learner.OEWCEstimatorLearner(dataset, self.learning_specs[i], self.run_config)
            single_learner.train()

            self.evaluate(i)

        return self.eval_matrix


class GroupCenterEWCLearner(GroupLearner):
    def __init__(self, set_of_dataset, learning_specs, n_task, run_config):
        super(GroupCenterEWCLearner, self).__init__(set_of_dataset, learning_specs, n_task, run_config)

    def base_train(self):
        base_dataset = self.set_of_dataset.list[0]
        base_learner = learner.CenterBaseEstimatorLearner(base_dataset, self.learning_specs[0], self.run_config, 0)
        base_learner.train()

        result = base_learner.evaluate()
        self.eval_matrix[0, 0] = result['accuracy']

    def train_and_evaluate(self):
        self.base_train()

        for i in range(1, self.n_task):
            dataset = self.set_of_dataset.list[i]
            single_learner = learner.CenterEWCEstimatorLearner(dataset, self.learning_specs[i], self.run_config, i)
            single_learner.train()

            self.evaluate(i)

        return self.eval_matrix


class GroupEWCLearner(GroupOEWCLearner):
    def __init__(self, set_of_dataset, learning_specs, n_task, run_config):
        super(GroupEWCLearner, self).__init__(set_of_dataset, learning_specs, n_task, run_config)

    def base_train(self):
        base_dataset = self.set_of_dataset.list[0]
        base_learner = learner.FullBaseEstimatorLearner(base_dataset, self.learning_specs[0], self.run_config, 0)
        base_learner.train()

        result = base_learner.evaluate()
        self.eval_matrix[0, 0] = result['accuracy']

    def train_and_evaluate(self):
        self.base_train()

        for i in range(1, self.n_task):
            dataset = self.set_of_dataset.list[i]
            single_learner = learner.OEWCEstimatorLearner(dataset, self.learning_specs[i], self.run_config, i)
            single_learner.train()

            self.evaluate(i)

        return self.eval_matrix


class GroupFedSGDLearner(GroupLearner):
    def __init__(self, set_of_dataset, learning_specs, n_task, run_config):
        super(GroupFedSGDLearner, self).__init__(set_of_dataset, learning_specs, n_task, run_config)
        self.n_fed_step = self.learning_specs[0].n_fed_step
        self.n_fed_round = self.learning_specs[0].n_fed_round
        self.n_fed_task = int(self.n_task * self.n_fed_round)

        print("n_fed_step:",  self.n_fed_step)
        print("n_fed_round:",  self.n_fed_round)
        print("n_fed_task:",  self.n_fed_task)

    def base_train(self):
        base_dataset = self.set_of_dataset.list[0]
        base_learner = learner.BaseEstimatorLearner(base_dataset, self.learning_specs[0], self.run_config)
        base_learner.train()

    def train_and_evaluate(self):
        for i in range(self.n_fed_task):
            i_task = i % 3
            Fed_learner = learner.SGDEstimatorLearner(self.set_of_dataset.list[i_task], self.learning_specs[0], self.run_config)
            Fed_learner.train()

        self.evaluate(self.n_task-1)

        return self.eval_matrix


class GroupFedOEWCLearner(GroupFedSGDLearner):
    def __init__(self, set_of_dataset, learning_specs, n_task, run_config):
        super(GroupFedOEWCLearner, self).__init__(set_of_dataset, learning_specs, n_task, run_config)

    def train_and_evaluate(self):
        self.base_train()

        for i in range(1, self.n_fed_task):
            i_task = i % 3
            dataset = self.set_of_dataset.list[i_task]
            single_learner = learner.OEWCEstimatorLearner(dataset, self.learning_specs[0], self.run_config)
            single_learner.train()

        self.evaluate(self.n_task - 1)

        return self.eval_matrix


class GroupFedQEWCLearner(GroupFedSGDLearner):
    def __init__(self, set_of_dataset, learning_specs, n_task, run_config):
        super(GroupFedQEWCLearner, self).__init__(set_of_dataset, learning_specs, n_task, run_config)

    def train_and_evaluate(self):
        self.base_train()

        for i in range(1, self.n_fed_task):
            i_task = i % 3
            dataset = self.set_of_dataset.list[i_task]
            single_learner = learner.QEWCEstimatorLearner(dataset, self.learning_specs[0], self.run_config)
            single_learner.train()

        self.evaluate(self.n_task - 1)

        return self.eval_matrix


class GroupInDepLearner(GroupLearner):
    def __init__(self, set_of_dataset, learning_specs, n_task, run_config):
        super(GroupInDepLearner, self).__init__(set_of_dataset, learning_specs, n_task, run_config)

    def train_and_evaluate(self):
        for i in range(self.n_task):
            dataset = self.set_of_dataset.list[i]
            single_learner = learner.SingleEstimatorLearner(dataset, self.learning_specs[i], self.run_config)
            single_learner.train()

            self.evaluate(i)

        return self.eval_matrix

    def evaluate(self, i):
        self.learning_specs[i].n_batch = 10
        eval_learner = learner.SingleEstimatorLearner(self.set_of_dataset.list[i], self.learning_specs[i],
                                                      self.run_config)
        result = eval_learner.evaluate()
        self.eval_matrix[i, i] = result['accuracy']


class GroupMultiLearner(GroupInDepLearner):
    def __init__(self, set_of_dataset, learning_specs, n_task, run_config):
        super(GroupMultiLearner, self).__init__(set_of_dataset, learning_specs, n_task, run_config)

    def train_and_evaluate(self):
        multi_learner = learner.MultiEstimatorLearner(self.set_of_dataset.concat(), self.learning_specs[0], self.run_config)
        multi_learner.train()

        for i in range(self.n_task):
            self.evaluate(i)

        return self.eval_matrix


class GroupIMMLearner(GroupLearner):
    def __init__(self, set_of_dataset, learning_specs, n_task, run_config):
        super(GroupIMMLearner, self).__init__(set_of_dataset, learning_specs, n_task, run_config)

    def train_and_evaluate(self):
        for i in range(self.n_task):
            dataset = self.set_of_dataset.list[i]
            imm_learner = learner.IMMEstimatorLearner(dataset, self.learning_specs[i], self.run_config, i)
            imm_learner.train()

            self.evaluate(i)

        return self.eval_matrix


class GroupHMTrainLearner(GroupLearner):
    def __init__(self, set_of_dataset, learning_specs, n_task, run_config, meta_learning_spec):
        super(GroupHMTrainLearner, self).__init__(set_of_dataset, learning_specs, n_task, run_config)
        self.meta_learning_spec = meta_learning_spec

    def base_train(self):
        base_dataset = self.set_of_dataset.list[0]
        base_learner = learner.MetaAlphaBaseEstimatorLearner(base_dataset, self.learning_specs[0], self.run_config, 0)
        base_learner.train()

    def train(self):
        self.base_train()

        for i in range(0, self.n_task):
            joint_dataset = self.set_of_dataset.list[i:i+2]
            meta_learner = learner.MetaAlphaTrainEstimatorLearner(joint_dataset, self.learning_specs[i],
                                                                  self.meta_learning_spec, self.run_config, i)
            meta_learner.train()


class GroupHMTestLearner(GroupLearner):
    def __init__(self, set_of_dataset, learning_specs, n_task, run_config, ws0, ws1):
        super(GroupHMTestLearner, self).__init__(set_of_dataset, learning_specs, n_task, run_config)
        self.ws0 = ws0
        self.ws1 = ws1

    def base_train(self):
        base_dataset = self.set_of_dataset.list[0]
        base_learner = learner.MetaAlphaWarmBaseEstimatorLearner(base_dataset, self.learning_specs[0], self.run_config, self.ws0, 0)
        base_learner.train()

        result = base_learner.evaluate()
        self.eval_matrix[0, 0] = result['accuracy']

    def train_and_evaluate(self):
        self.base_train()

        for i in range(1, self.n_task):
            dataset = self.set_of_dataset.list[i]
            meta_learner = learner.MetaAlphaWarmTestEstimatorLearner(dataset, self.learning_specs[i], self.run_config, self.ws1, i)
            meta_learner.train()

            self.evaluate(i)

        return self.eval_matrix

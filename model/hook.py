import tensorflow as tf
import numpy as np


class GradientHook(tf.train.SessionRunHook):
    def __init__(self, grad_and_var, n_batch):
        self.gradients = grad_and_var[0]
        self.variable = grad_and_var[1]
        self.name = self.variable.name[5:-2]
        self.n_batch = n_batch

    def begin(self):
        self.global_step = tf.train.get_global_step()
        self.condition = tf.equal(self.global_step % 600, 0)

    def after_run(self, run_context, run_values):
        _ = run_context
        self.save_fisher_component(run_values.results)


class AverageAssignGradientHook(GradientHook):
    def __init__(self, grad_and_var, n_batch):
        super(AverageAssignGradientHook, self).__init__(grad_and_var, n_batch)

    def begin(self):
        super(AverageAssignGradientHook, self).begin()

        self.avg_gradients = tf.Variable(tf.zeros_like(self.gradients), name=('avg/' + self.name))
        self.avg_gradients = self.avg_gradients.assign(self.gradients)
        self.zero_avg_gradients = tf.where(self.condition, tf.zeros_like(self.gradients), self.avg_gradients)
        self.avg_gradients = tf.assign(self.avg_gradients, self.zero_avg_gradients)

    def save_fisher_component(self, results):
        if results['global_step'] % 100 == 0:
            print(self.name, ': avg', np.linalg.norm(results['avg_gradients']))
            print('step condtion: ', results['condition'])

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({'avg_gradients': self.avg_gradients,
                                        'global_step': self.global_step,
                                        'condition': self.condition})


class SquareAccumulationGradientHook(GradientHook):
    def __init__(self, grad_and_var, n_batch):
        super(SquareAccumulationGradientHook, self).__init__(grad_and_var, n_batch)

    def begin(self):
        super(SquareAccumulationGradientHook, self).begin()
        self.sum_gradients = tf.Variable(tf.zeros_like(self.gradients), name=('fisher/' + self.name))
        self.period = int(60000/self.n_batch)
        self.assign_condition = tf.greater_equal(self.global_step % self.period, self.period-self.n_batch)
        self.assign_gradients = tf.where(self.assign_condition, tf.math.square(self.gradients), tf.zeros_like(self.gradients))
        self.sum_gradients = tf.assign(self.sum_gradients, self.assign_gradients)

    def save_fisher_component(self, results):
        if (results['global_step']+self.n_batch) % 6000 == 0:
            print(self.name, ': fisher', np.linalg.norm(results['sum_gradients']))
            print('step condtion: ', results['condition'])

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({'sum_gradients': self.sum_gradients,
                                        'global_step': self.global_step,
                                        'condition': self.assign_condition})


class SequentialSquareAccumulationGradientHook(GradientHook):
    def __init__(self, grad_and_var, n_batch, n_task):
        super(SequentialSquareAccumulationGradientHook, self).__init__(grad_and_var, n_batch)
        self.n_task = n_task
        self.fisher = []
        self.theta = []
        self.conditions = []

    def begin(self):
        super(SequentialSquareAccumulationGradientHook, self).begin()

        for i in range(self.n_task):
            self.conditions.append(tf.equal(self.global_step, int(60000/self.n_batch) * i))
            self.fisher.append(tf.Variable(tf.zeros_like(self.gradients), name=('fisher' + str(i) + '/' + self.name)))
            self.theta.append(tf.Variable(tf.zeros_like(self.gradients), name=('theta' + str(i) + '/' + self.name)))

        self.sum_gradients = tf.Variable(tf.zeros_like(self.gradients), name=('fisher/' + self.name))
        self.sum_gradients = self.sum_gradients.assign_add(tf.math.square(self.gradients))

        for i in range(self.n_task):
            self.fisher_cond = tf.where(self.conditions[i], self.sum_gradients, self.fisher[i])
            self.theta_cond = tf.where(self.conditions[i], self.variable, self.theta[i])

            self.fisher[i] = tf.assign(self.fisher[i], self.fisher_cond)
            self.theta[i] = tf.assign(self.theta[i], self.theta_cond)

    def save_fisher_component(self, results):
        if results['global_step'] % 600 == 0:
            print(self.name, ': fisher', np.linalg.norm(results['sum_gradients']))

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({'sum_gradients': self.sum_gradients,
                                        'global_step': self.global_step})

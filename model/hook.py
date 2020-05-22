import tensorflow as tf
import numpy as np


class GradientHook(tf.train.SessionRunHook):
    def __init__(self, grad_and_var, n_batch):
        self.gradients = grad_and_var[0]
        self.variable = grad_and_var[1]
        self.name = self.variable.name[5:-2]
        self.n_batch = n_batch
        self.period = int(60000 / self.n_batch)

        self.global_step = tf.train.get_global_step()
        self.condition = tf.equal(self.global_step % 600, 0)

    def after_run(self, run_context, run_values):
        _ = run_context
        self.save_fisher_component(run_values.results)

    def save_fisher_component(self, results):
        if (results['global_step']+self.n_batch) % self.period == 0:
            print(self.name, ': fisher', np.linalg.norm(results['sum_gradients']))
            print('step condtion: ', results['condition'])


class SquareAccumulationGradientHook(GradientHook):
    def __init__(self, grad_and_var, n_batch):
        super(SquareAccumulationGradientHook, self).__init__(grad_and_var, n_batch)

    def begin(self):
        self.sum_gradients = tf.Variable(tf.zeros_like(self.gradients), name=('fisher/' + self.name))
        self.assign_condition = tf.greater_equal(self.global_step % self.period, self.period - self.n_batch)
        self.assign_gradients = tf.where(self.assign_condition, tf.math.square(self.gradients), tf.zeros_like(self.gradients))
        self.sum_gradients = tf.assign_add(self.sum_gradients, self.assign_gradients)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({'sum_gradients': self.sum_gradients,
                                        'global_step': self.global_step,
                                        'condition': self.assign_condition})


class CenterSquareAccumulationGradientHook(SquareAccumulationGradientHook):
    def __init__(self, grad_and_var, n_batch):
        super(CenterSquareAccumulationGradientHook, self).__init__(grad_and_var, n_batch)

    def begin(self):
        self.sum_gradients = tf.Variable(tf.zeros_like(self.gradients), name=('fisher/' + self.name))
        self.assign_condition = tf.greater_equal(self.global_step % self.period, 0)
        self.assign_gradients = tf.where(self.assign_condition, tf.math.square(self.gradients),
                                         tf.zeros_like(self.gradients))

        self.sum_gradients = tf.assign_add(self.sum_gradients, self.assign_gradients)

        self.sum_thetas = tf.Variable(tf.zeros_like(self.variable), name=('center/' + self.name))
        self.assign_thetas = tf.where(self.assign_condition, self.variable, tf.zeros_like(self.variable))
        self.sum_thetas = tf.assign_add(self.sum_thetas, self.assign_thetas)

    def save_fisher_component(self, results):
        if (results['global_step']+self.n_batch) % self.period == 0:
            print(self.name, ': fisher', np.linalg.norm(results['sum_gradients']))
            print(self.name, ': center', np.linalg.norm(results['sum_thetas']))
            print('step condtion: ', results['condition'])

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({'sum_gradients': self.sum_gradients,
                                        'sum_thetas': self.sum_thetas,
                                        'global_step': self.global_step,
                                        'condition': self.assign_condition})


class SequentialSquareAccumulationGradientHook(SquareAccumulationGradientHook):
    def __init__(self, grad_and_var, n_batch, n_task, i_task):
        super(SequentialSquareAccumulationGradientHook, self).__init__(grad_and_var, n_batch)
        self.n_task = n_task
        self.i_task = i_task
        self.fisher = []
        self.theta = []

        for i in range(self.n_task):
            self.fisher.append(tf.Variable(tf.zeros_like(self.gradients), name=('fisher' + str(i) + '/' + self.name)))
            self.theta.append(tf.Variable(tf.zeros_like(self.gradients), name=('theta' + str(i) + '/' + self.name)))

    def begin(self):
        self.condition = tf.greater_equal(self.global_step % self.period, self.period - self.n_batch)
        self.assigned_fisher = tf.where(self.condition, tf.math.square(self.gradients), tf.zeros_like(self.gradients))
        self.assigned_theta = tf.where(self.condition, tf.math.square(self.gradients), tf.zeros_like(self.gradients))

        self.fisher[self.i_task] = tf.assign_add(self.fisher[self.i_task], self.assigned_fisher)
        self.theta[self.i_task] = tf.assign_add(self.theta[self.i_task], self.assigned_theta)

    def save_fisher_component(self, results):
        if (results['global_step'] + self.n_batch) % self.period == 0:
            print(self.name, ': fisher', np.linalg.norm(results['fisher']))
            print(self.name, ': theta', np.linalg.norm(results['theta']))

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({'fisher': self.fisher,
                                        'theta': self.theta,
                                        'global_step': self.global_step})

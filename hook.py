import tensorflow as tf
import numpy as np


class GradientHook(tf.train.SessionRunHook):
    def __init__(self, grad_and_var):
        self.gradients = grad_and_var[0]
        self.variable = grad_and_var[1]
        self.name = self.variable.name[:-2]


class AverageAssignGradientHook(GradientHook):
    def __init__(self, grad_and_var):
        super(GradientHook, self).__init__(grad_and_var)

    def begin(self):
        self.global_step = tf.train.get_global_step()
        self.condition = tf.equal(self.global_step % 6000, 0)
        self.avg_gradients = tf.Variable(tf.zeros_like(self.gradients), name=('avg/' + self.name))
        self.avg_gradients = self.avg_gradients.assign(self.gradients)
        self.zero_avg_gradients = tf.where(self.condition, tf.zeros_like(self.gradients), self.avg_gradients)
        self.avg_gradients = tf.assign(self.avg_gradients, self.zero_avg_gradients)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({'avg_gradients': self.avg_gradients,
                                        'global_step': self.global_step,
                                        'condition': self.condition})

    def save_fisher_component(self, results):
        if results['global_step'] % 1000 == 0:
            print(self.name, ': avg', np.linalg.norm(results['avg_gradients']))
            print('step condtion: ', results['condition'])

    def after_run(self, run_context, run_values):
        _ = run_context
        self.save_fisher_component(run_values.results)


class SquareAccumulationGradientHook(GradientHook):
    def __init__(self, grad_and_var):
        super(SquareAccumulationGradientHook, self).__init__(grad_and_var)

    def begin(self):
        self.global_step = tf.train.get_global_step()
        self.sum_gradients = tf.Variable(tf.zeros_like(self.gradients), name=('fisher/' + self.name))
        self.sum_gradients = self.sum_gradients.assign_add(tf.math.square(self.gradients))
        self.condition = tf.equal(self.global_step % 6000, 0)
        self.zero_gradients = tf.where(self.condition, tf.zeros_like(self.gradients), self.sum_gradients)
        self.sum_gradients = tf.assign(self.sum_gradients, self.zero_gradients)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({'sum_gradients': self.sum_gradients,
                                        'global_step': self.global_step,
                                        'condition': self.condition})

    def save_fisher_component(self, results):
        if results['global_step'] % 1000 == 0:
            print(self.name, ': fisher', np.linalg.norm(results['sum_gradients']))
            print('step condtion: ', results['condition'])

    def after_run(self, run_context, run_values):
        _ = run_context
        self.save_fisher_component(run_values.results)

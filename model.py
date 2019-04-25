import numpy as np
import tensorflow as tf


def single(features, labels, mode, params):

    model = FCN("main")
    logits = model(features)
    predictions = tf.argmax(logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        softmax_layer = tf.keras.layers.Softmax()
        probabilities = softmax_layer(logits)
        return tf.estimator.EstimatorSpec(mode, predictions={'predictions': predictions, 'probabilities': probabilities})

    one_hot_labels = tf.one_hot(labels, 10)
    loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels, predictions)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'accuracy': accuracy})

    opt = tf.train.GradientDescentOptimizer(learning_rate=params['lr'])
    grads_and_vars = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def base(features, labels, mode, params):

    model = FCN("main")
    logits = model(features)
    predictions = tf.argmax(logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        softmax_layer = tf.keras.layers.Softmax()
        probabilities = softmax_layer(logits)
        return tf.estimator.EstimatorSpec(mode, predictions={'predictions': predictions, 'probabilities': probabilities})

    one_hot_labels = tf.one_hot(labels, 10)
    loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels, predictions)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'accuracy': accuracy})

    opt = tf.train.GradientDescentOptimizer(learning_rate=params['lr'])
    grads_and_vars = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

    fisher_hook = []
    for grad_and_var in grads_and_vars:
        fisher_hook.append(GradientSquareAccumulationHook(grad_and_var))

    return tf.estimator.EstimatorSpec(mode, loss=loss, training_hooks=fisher_hook, train_op=train_op)


def ewc(features, labels, mode, params):

    model = FCN()
    logits = model(features)
    predictions = tf.argmax(logits, axis=1)
    one_hot_labels = tf.one_hot(labels, 10)

    loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels, predictions)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'accuracy': accuracy})

    opt = tf.train.GradientDescentOptimizer(learning_rate=params['lr'])

    checkpoint = params['eval_model_dir']
    ewc_loss = compute_ewc_loss(model, checkpoint)

    loss = loss + params['alpha'] * ewc_loss

    grads_and_vars = opt.compute_gradients(loss, var_list=model.weights)
    train_op = opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

    fisher_hook = []
    for grad_and_var in grads_and_vars:
        fisher_hook.append(GradientSquareAccumulationHook(grad_and_var))

    return tf.estimator.EstimatorSpec(mode, loss=loss, training_hooks=fisher_hook, train_op=train_op)


def meta_base(features, labels, mode, params):

    model = FCN("main")
    logits = model(features)
    predictions = tf.argmax(logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        softmax_layer = tf.keras.layers.Softmax()
        probabilities = softmax_layer(logits)
        return tf.estimator.EstimatorSpec(mode, predictions={'predictions': predictions, 'probabilities': probabilities})

    one_hot_labels = tf.one_hot(labels, 10)
    loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels, predictions)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'accuracy': accuracy})

    opt = tf.train.GradientDescentOptimizer(learning_rate=params['lr'])

    with tf.variable_scope("meta"):
        meta_model = MetaFCN()

    grads_and_vars = opt.compute_gradients(loss,var_list=model.weights)
    train_op = opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

    fisher_hook = []
    for grad_and_var in grads_and_vars:
        fisher_hook.append(GradientSquareAccumulationHook(grad_and_var))

    return tf.estimator.EstimatorSpec(mode, loss=loss, training_hooks=fisher_hook, train_op=train_op)


def meta_ewc(features, labels, mode, params):
    model = FCN()
    logits = model(features)
    predictions = tf.argmax(logits, axis=1)
    one_hot_labels = tf.one_hot(labels, 10)

    loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels, predictions)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'accuracy': accuracy})

    opt = tf.train.GradientDescentOptimizer(learning_rate=params['lr'])

    with tf.variable_scope("meta"):
        meta_model = MetaFCN()

    grads_and_vars = opt.compute_gradients(loss, var_list=model.weights)
    cur_grads, cur_vars = zip(*grads_and_vars)

    train_op = opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

    checkpoint = params['model']
    fisher_grads = load_tensors(model, checkpoint, "fisher/")
    pre_grads = load_tensors(model, checkpoint, "avg/")
    pre_model_weights = load_tensors(model, checkpoint, "")

    meta_output = meta_model(fisher_grads)

    meta_label = compute_importance(cur_grads, pre_grads, model.weights, pre_model_weights)

    meta_loss = tf.losses.mean_squared_error(meta_output, meta_label)
    tf.summary.scalar(name="meta_loss", tensor=meta_loss)
    meta_opt = tf.train.GradientDescentOptimizer(learning_rate=params['meta_lr'])

    meta_grads_and_vars = meta_opt.compute_gradients(meta_loss, var_list=meta_model.weights)
    meta_train_op = opt.apply_gradients(meta_grads_and_vars, global_step=tf.train.get_global_step())

    fisher_hook = []
    for grad_and_var in grads_and_vars:
        fisher_hook.append(GradientSquareAccumulationHook(grad_and_var))

    return tf.estimator.EstimatorSpec(mode, loss=meta_loss, training_hooks=fisher_hook,
                                      train_op=tf.group([train_op, meta_train_op]))


def meta_eval(features, labels, mode, params):
    model = FCN()
    logits = model(features)
    predictions = tf.argmax(logits, axis=1)
    one_hot_labels = tf.one_hot(labels, 10)

    loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels, predictions)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'accuracy': accuracy})

    opt = tf.train.GradientDescentOptimizer(learning_rate=params['lr'])

    with tf.variable_scope("meta"):
        meta_model = MetaFCN()

    checkpoint = params['eval_model_dir']
    meta_loss = compute_meta_loss(model, meta_model, checkpoint)

    loss = loss + params['alpha'] * meta_loss

    grads_and_vars = opt.compute_gradients(loss, var_list=model.weights)
    train_op = opt.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

    fisher_hook = []
    for grad_and_var in grads_and_vars:
        fisher_hook.append(GradientSquareAccumulationHook(grad_and_var))

    return tf.estimator.EstimatorSpec(mode, loss=loss, training_hooks=fisher_hook, train_op=train_op)


def load_tensors(cur_model, checkpoint, surfix):
    tensor_list = []
    for weight in cur_model.weights:
        name = weight.name[:-2]
        loaded_tensor = tf.train.load_variable(checkpoint, surfix+name)
        tensor_list.append(loaded_tensor)

    return tensor_list


def compute_meta_loss(cur_model, meta_model, checkpoint):
    meta_loss = 0
    for weight in cur_model.weights:
        shape = weight.shape
        name = weight.name[:-2]
        cur_var = weight
        pre_var = tf.train.load_variable(checkpoint, name)
        fisher = tf.train.load_variable(checkpoint, 'fisher/'+name)
        meta_output = meta_model(fisher)
        meta_output = tf.reshape(meta_output, shape=shape)

        meta_loss = meta_loss + tf.losses.mean_squared_error(cur_var, pre_var, meta_output)

    return meta_loss


def compute_ewc_loss(cur_model, checkpoint):
    ewc_loss = 0
    for weight in cur_model.weights:
        name = weight.name[:-2]
        cur_var = weight
        pre_var = tf.train.load_variable(checkpoint, name)
        fisher = tf.train.load_variable(checkpoint, 'fisher/'+name)

        ewc_loss = ewc_loss + tf.losses.mean_squared_error(cur_var, pre_var, fisher)

    return ewc_loss


def compute_importance(cur_grads, pre_grads, cur_theta, pre_theta):
    flat_cur_grads = layer_to_flat(cur_grads)
    flat_pre_grads = layer_to_flat(pre_grads)
    flat_cur_theta = layer_to_flat(cur_theta)
    flat_pre_theta = layer_to_flat(pre_theta)

    eps = tf.math.scalar_mul(1e-6, tf.ones_like(flat_pre_grads))

    importance = (flat_cur_grads-flat_pre_grads)/(2*(flat_cur_theta-flat_pre_theta) + eps)

    importance = tf.clip_by_value(importance, clip_value_min=0.0, clip_value_max=10.0)

    return importance


def layer_to_flat(grads):
    grad_list = []
    for grad in grads:
        flat_grad = tf.reshape(grad, [-1, 1])
        grad_list.append(flat_grad)

    return tf.concat(grad_list, axis=0)


def flat_to_layer(grads, cur_vars):
    grads_and_vars = []
    for var in cur_vars:
        pre_shape = tf.shape(var)
        flat_var = tf.reshape(var, [-1, 1])
        size = tf.shape(flat_var)[0]
        begin = 0
        grad_vec = tf.reshape(grads, [-1])
        flat_grad = tf.slice(grad_vec, [begin], [size])
        grad = tf.reshape(flat_grad, pre_shape)
        grads_and_vars.append((grad, var))
        begin = begin + size

    return grads_and_vars


class FCN(tf.keras.models.Model):
    def __init__(self, *args):
        super(FCN, self).__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.InputLayer((784,)),
            tf.keras.layers.Dense(100, activation='relu', name='dense1'),
            tf.keras.layers.Dense(100, activation='relu', name='dense2'),
            tf.keras.layers.Dense(10, name='dense3')])

    def call(self, inputs):
        return self.net(inputs)


class MetaFCN(tf.keras.models.Model):
    def __init__(self):
        super(MetaFCN, self).__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.InputLayer((1,)),
            tf.keras.layers.Dense(50, activation='relu', name='dense1'),
            tf.keras.layers.Dense(50, activation='relu', name='dense2'),
            tf.keras.layers.Dense(50, activation='relu', name='dense3'),
            tf.keras.layers.Dense(1, name='dense4')])

    def call(self, inputs):
        return self.net(layer_to_flat(inputs))


class GradientSquareAccumulationHook(tf.train.SessionRunHook):
    def __init__(self, grad_and_var):
        self.gradients = grad_and_var[0]
        self.variable = grad_and_var[1]
        self.name = self.variable.name[:-2]

    def begin(self):
        self.global_step = tf.train.get_global_step()
        self.sum_gradients = tf.Variable(tf.zeros_like(self.gradients), name=('fisher/' + self.name))
        self.sum_gradients = self.sum_gradients.assign_add(tf.math.square(self.gradients))
        self.condition = tf.equal(self.global_step % 6000, 0)
        self.zero_gradients = tf.where(self.condition, tf.zeros_like(self.gradients), self.sum_gradients)
        self.sum_gradients = tf.assign(self.sum_gradients, self.zero_gradients)

        self.avg_gradients = tf.Variable(tf.zeros_like(self.gradients), name=('avg/' + self.name))
        self.avg_gradients = self.avg_gradients.assign(self.gradients)
        self.zero_avg_gradients = tf.where(self.condition, tf.zeros_like(self.gradients), self.avg_gradients)
        self.avg_gradients = tf.assign(self.avg_gradients, self.zero_avg_gradients)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({'sum_gradients': self.sum_gradients,
                                        'avg_gradients': self.avg_gradients,
                                        'global_step': self.global_step,
                                        'condition': self.condition})

    def save_fisher_component(self, results):
        if results['global_step'] % 1000 == 0:
            print(self.name, ': fisher', np.linalg.norm(results['sum_gradients']))
            print(self.name, ': avg', np.linalg.norm(results['avg_gradients']))
            print('step condtion: ', results['condition'])

    def after_run(self, run_context, run_values):
        _ = run_context
        self.save_fisher_component(run_values.results)

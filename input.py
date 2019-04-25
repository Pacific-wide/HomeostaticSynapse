import tensorflow as tf
import dataset as mnist


def train_input_fn(x_train, y_train, n_epoch, n_batch, p):
    d_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    perm_train = d_train.map(lambda x, y: (mnist.permute(x, p), y))
    perm_train = perm_train.repeat(n_epoch).batch(n_batch)

    return perm_train


def eval_input_fn(x_eval, y_eval, n_batch, p):
    d_eval = tf.data.Dataset.from_tensor_slices((x_eval, y_eval))
    perm_eval = d_eval.map(lambda x, y: (mnist.permute(x, p), y))
    perm_eval = perm_eval.batch(n_batch)

    return perm_eval


def train_multi_input_fn(x_train, y_train, n_epoch, n_batch, ps):
    d_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    n_task = ps.shape[0]

    for i, p in enumerate(ps):
        perm_train = d_train.map(lambda x, y: (mnist.permute(x, p), y))
        if i == 0:
            comb_train = perm_train
        else:
            comb_train = comb_train.concatenate(perm_train)

    comb_train = comb_train.shuffle(60000*n_task).repeat(n_epoch).batch(n_batch)

    return comb_train

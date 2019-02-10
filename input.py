import tensorflow as tf
import dataset as mnist


def train_input_fn(n_epoch, n_batch, p):
    d_train, _ = mnist.load_mnist_datasets()
    perm_d_train = d_train.map(lambda feature, label: (mnist.permute(feature, p), label))
    perm_d_train = perm_d_train.repeat(n_epoch).batch(n_batch)

    return perm_d_train


def eval_input_fn(n_batch, p):
    _, d_eval = mnist.load_mnist_datasets()
    perm_d_eval = d_eval.map(lambda feature, label: (mnist.permute(feature, p), label))
    perm_d_eval = perm_d_eval.batch(n_batch)

    return perm_d_eval


def meta_train_input_fn(n_epoch, n_batch, ps):
    d_train, _ = mnist.load_mnist_datasets()
    n_task = ps.shape[0]
    task_tuple = ()
    for p in ps:
        perm_d_train = d_train.map(lambda feature, label: (mnist.permute(feature, p), label))
        task_tuple = task_tuple + (perm_d_train,)

    comb_d_train = tf.data.Dataset.zip(task_tuple)
    flat_d_train = comb_d_train.flat_map(map_fn)
    data_tuple = (flat_d_train, ) + task_tuple

    final_train = tf.data.Dataset.zip(data_tuple)
    final_train = final_train.map(unfold_tuple)
    final_train = final_train.repeat(n_epoch).batch(2*n_task*n_batch)

    return final_train


def aligned_multi_train_input_fn(n_epoch, n_batch, ps):
    d_train, _ = mnist.load_mnist_datasets()
    n_task = ps.shape[0]
    task_tuple = ()
    for p in ps:
        perm_d_train = d_train.map(lambda feature, label: (mnist.permute(feature, p), label))
        task_tuple = task_tuple + (perm_d_train, )

    comb_d_train = tf.data.Dataset.zip(task_tuple)
    flat_d_train = comb_d_train.flat_map(map_fn)
    flat_d_train = flat_d_train.repeat(n_epoch).batch(n_task*n_batch)

    return flat_d_train


def shuffled_multi_train_input_fn(n_epoch, n_batch, ps):
    d_train, _ = mnist.load_mnist_datasets()
    n_task = ps.shape[0]

    for i, p in enumerate(ps):
        perm_d_train = d_train.map(lambda feature, label: (mnist.permute(feature, p), label))
        if i == 0:
            comb_d_train = perm_d_train
        comb_d_train = comb_d_train.concatenate(perm_d_train)

    comb_d_train = comb_d_train.repeat(n_epoch).batch(n_task*n_batch)

    return comb_d_train


def unfold_tuple(*x):
    t1, t2, t3 = x

    f1, l1 = t1
    f2, l2 = t2
    f3, l3 = t3

    return (f1, f2, f3), (l1, l2, l3)


def map_fn(*x):
    features, labels = zip(*x)
    return tf.data.Dataset.from_tensor_slices((tf.stack(features), tf.stack(labels)))

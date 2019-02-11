import tensorflow as tf
import dataset as mnist


def train_input_fn(n_epoch, n_batch, p):
    train, _ = mnist.load_mnist_datasets()
    perm_train = train.map(lambda x, y: (mnist.permute(x, p), y))
    perm_train = perm_train.repeat(n_epoch).batch(n_batch)

    return perm_train


def eval_input_fn(n_batch, p):
    _, eval = mnist.load_mnist_datasets()
    perm_eval = eval.map(lambda x, y: (mnist.permute(x, p), y))
    perm_eval = perm_eval.batch(n_batch)

    return perm_eval


def meta_eval_input_fn(n_epoch, n_batch, ps):
    train, _ = mnist.load_mnist_datasets()

    for i, p in enumerate(ps):
        perm_train = train.map(lambda x, y: (mnist.permute(x, p), y))
        if i == 0:
            concat_train = perm_train
        else:
            concat_train = concat_train.concatenate(perm_train)

    concat_train = concat_train.repeat(n_epoch).batch(n_batch)
    print(concat_train.output_shapes)

    return concat_train


def meta_train_input_fn(n_epoch, n_batch, ps):
    train, _ = mnist.load_mnist_datasets()
    n_task = ps.shape[0]
    task_tuple = ()
    for p in ps:
        perm_train = train.map(lambda x, y: (mnist.permute(x, p), y))
        task_tuple = task_tuple + (perm_train,)

    comb_train = tf.data.Dataset.zip(task_tuple)
    flat_train = comb_train.flat_map(map_fn)
    data_tuple = (flat_train, ) + task_tuple

    final_train = tf.data.Dataset.zip(data_tuple)
    final_train = final_train.map(unfold_tuple)
    final_train = final_train.repeat(n_epoch).batch(2*n_task*n_batch)

    return final_train


def aligned_multi_train_input_fn(n_epoch, n_batch, ps):
    train, _ = mnist.load_mnist_datasets()
    n_task = ps.shape[0]
    task_tuple = ()
    for p in ps:
        perm_train = train.map(lambda x, y: (mnist.permute(x, p), y))
        task_tuple = task_tuple + (perm_train, )

    comb_train = tf.data.Dataset.zip(task_tuple)
    flat_train = comb_train.flat_map(map_fn)
    flat_train = flat_train.repeat(n_epoch).batch(n_task*n_batch)

    return flat_train


def shuffled_multi_train_input_fn(n_epoch, n_batch, ps):
    train, _ = mnist.load_mnist_datasets()
    n_task = ps.shape[0]

    for i, p in enumerate(ps):
        perm_train = train.map(lambda x, y: (mnist.permute(x, p), y))
        task_tuple = task_tuple + (perm_train,)
        if i == 0:
            comb_train = perm_train
        else:
            comb_train = comb_train.concatenate(perm_train)

    comb_train = comb_train.repeat(n_epoch).batch(n_task*n_batch)

    return comb_train


def unfold_tuple(*x):
    t1, t2, t3 = x

    f1, l1 = t1
    f2, l2 = t2
    f3, l3 = t3

    return (f1, f2, f3), (l1, l2, l3)


def map_fn(*z):
    x, y = zip(*z)
    return tf.data.Dataset.from_tensor_slices((tf.stack(x), tf.stack(y)))

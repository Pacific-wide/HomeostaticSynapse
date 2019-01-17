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

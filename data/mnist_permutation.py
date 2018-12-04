import random
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('type', 'PERM', 'type of dataset')
flags.DEFINE_integer('n_tasks', 3, 'number of tasks')
flags.DEFINE_integer('n_types', 3, 'number of types')


train, test = tf.keras.datasets.mnist.load_data()
mnist_x_tr, mnist_y_tr = train
mnist_x_te, mnist_y_te = test

mnist_x_tr = mnist_x_tr.reshape(mnist_x_tr.shape[0],-1) / 255.0
mnist_x_te = mnist_x_te.reshape(mnist_x_te.shape[0],-1) / 255.0

for i in range(FLAGS.n_types):

    x_tr = np.zeros(mnist_x_tr.shape + (FLAGS.n_tasks,)) # (batch, pixels, n_tasks)
    y_tr = np.zeros(mnist_y_tr.shape + (FLAGS.n_tasks,)) # (batch, n_tasks)
    x_te = np.zeros(mnist_x_te.shape + (FLAGS.n_tasks,))
    y_te = np.zeros(mnist_y_te.shape + (FLAGS.n_tasks,))

    for j in range(FLAGS.n_tasks):
        p = np.random.permutation(x_tr.shape[1])

        x_tr[:,:,j] = mnist_x_tr[:,p]
        y_tr[:,j] =  mnist_y_tr
        x_te[:,:,j] = mnist_x_te[:,p]
        y_te[:,j] =  mnist_y_te

    np.savez_compressed('dataset/mnist_perm'+' '+str(i), x_tr = x_tr, y_tr = y_tr, \
                                      x_te = x_te, y_te = y_te)





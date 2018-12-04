import random
import numpy as np
import tensorflow as tf
from PIL import Image

def rotate(data, angle):
    data = data.reshape(-1,28,28)
    shape = data.shape
    result = np.zeros(shape)
    
    for i in range(shape[0]):
        img = Image.fromarray(data[i], mode='L')
        result[i] = img.rotate(angle)
    
    result = result.reshape(shape[0],-1) / 255.0

    return result

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('type','ROTA','type of dataset')
flags.DEFINE_integer('n_tasks', 3, 'number of tasks')
flags.DEFINE_integer('n_types', 3, 'number of types')
flags.DEFINE_integer('max_angle', 180, 'maximum angle')

train, test = tf.keras.datasets.mnist.load_data()
mnist_x_tr, mnist_y_tr = train
mnist_x_te, mnist_y_te = test

mnist_x_tr = mnist_x_tr.reshape(mnist_x_tr.shape[0],-1)
mnist_x_te = mnist_x_te.reshape(mnist_x_te.shape[0],-1)

for i in range(FLAGS.n_types):

    x_tr = np.zeros(mnist_x_tr.shape + (FLAGS.n_tasks,)) # (batch, pixels, n_tasks)
    y_tr = np.zeros(mnist_y_tr.shape + (FLAGS.n_tasks,)) # (batch, n_tasks)
    x_te = np.zeros(mnist_x_te.shape + (FLAGS.n_tasks,))
    y_te = np.zeros(mnist_y_te.shape + (FLAGS.n_tasks,))

    for j in range(FLAGS.n_tasks):
        angle = random.random() * FLAGS.max_angle

        x_tr[:,:,j] = rotate(mnist_x_tr, angle)
        y_tr[:,j] =  mnist_y_tr
        x_te[:,:,j] = rotate(mnist_x_te, angle)
        y_te[:,j] =  mnist_y_te

    np.savez_compressed('dataset/mnist_rota'+' '+str(i), x_tr = x_tr, y_tr = y_tr, \
                                                         x_te = x_te, y_te = y_te)
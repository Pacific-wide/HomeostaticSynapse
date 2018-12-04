import importlib
import datetime
import random
import uuid
import time
import os

import numpy as np

import tensorflow as tf

def load_datasets(FLAGS):
    dataset = np.load(FLAGS.data_path + FLAGS.type + ' 0.npz')
    x_tr = dataset['x_tr']
    y_tr = dataset['y_tr']
    x_te = dataset['x_te']
    y_te = dataset['y_te']
    
    return x_tr, y_tr, x_te, y_te
    

if __name__ == "__main__":
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string('type', 'perm', 'type of dataset')
    flags.DEFINE_integer('n_tasks', 3, 'number of tasks') 
    flags.DEFINE_integer('n_types', 3, 'number of types') 
    
    # main model parameters
    flags.DEFINE_string('model', 'single', 'model to train')
    flags.DEFINE_integer('n_hiddens', 100, 'number of hidden neurons at each layer') 
    flags.DEFINE_integer('n_layers', 2, 'number of hidden layers') 
    
    # main optimizer parameters
    flags.DEFINE_integer('n_epochs', 1,'Number of epochs per task')
    flags.DEFINE_integer('batch_size', 1,'batch size')
    flags.DEFINE_float('lr', 1e-3, 'SGD learning rate')

    # experiment parameters
    flags.DEFINE_string('save_path', 'results/', 'save models at the end of training')

    # data parameters
    flags.DEFINE_string('data_path', 'data/dataset/', 'path where data is located')

    flags.DEFINE_integer('samples_per_task', 1000,'training samples per task')
    

    # unique identifier
    uid = uuid.uuid4().hex

    # load data
    x_tr, y_tr, x_te, y_te = load_datasets(FLAGS)

    # set up DataSet

    
    # load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)
    
    # run model on continuum
    result_t, result_a, spent_time = life_experience(model, continuum, x_te, args)


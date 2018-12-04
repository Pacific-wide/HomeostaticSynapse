#!/bin/bash

MY_PYTHON="python"

cd data/dataset

rm -fr *

cd ..


$MY_PYTHON mnist_permutation.py --n_tasks 3
#$MY_PYTHON mnist_rotation.py --n_tasks 3
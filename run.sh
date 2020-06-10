#!/bin/bash

PYTHON="python"


# Evaluate comparison group


# Alternative Training

MODEL="Single" // OEWC, EWC, Indep, Multi, IMM


for seed in {0..9}
do
    // Check Argument Information in train.py
    $PYTHON train.py --seed $seed --model $MODEL
    rm -r $MODEL
done




# Meta Training (Homeostatic Model)

$PYTHON meta_alpha_train.py --seed 12 --n_task 30 //  (NOTICE, seed > 9)


 # Meta Testing (with trained Homeostatic Model)

$PYTHON meta_alpha_test.py --seed 0
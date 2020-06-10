

# Title

This repository is the official implementation of Homeostasis-Inspired Continual Learning: Learning to Control Structural Regularization.


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> It is recommended to use the Anaconda

## Training Manually

To train the Homeostatic Meta-Model (HM) in the paper, run this command:

```train
python meta_alpha_train.py --data MNISTBPERM --n_task 30 --seed 20
```

or Execute the pre-established shell script:

```
./run.sh
```

## Evaluation

To evaluate Homeostatic Meta-Model on MNIST-BPERM, run:

```eval
python meta_alpha_test.py --data MNISTPERM
```

> Using homeostatic meta-trained model, you can evaluate the performance on continual learning.

## Alternative Models

You can also evaluate alternative models for comparison

- Single : a single learner based on SGD for a sequence of tasks
- InDep : a dedicated (independent) learner based on SGD for each task
- EWC : Elastic Weight Consolidation (regularized with the dedicated Fisher information for each task) [1]
- OEWC : Online Elastic Weight Consolidation (regularized with the accumulated Fisher information) [2]
- IMM  : Incremental Moment Matching with a weight transfer method [3]
- Multi : Multi-task learning (allowed to access all the tasks, violation of strict CL scenario)

```train
python train.py --model InDep --data MNISTBPERM --n_task 30 --seed 0 
```

> Use "--model" argument with above model names to train other alternatives

## Results

Our model achieves the following performance on the sequence of 10 MNIST-PERM tasks:

### MNIST-PERM Dataset

| Model   | Average Accuracy | Average Forgetting |
| --------|------------------| ------------------ |
| Single  |   59.62% +- 2.9  |    37.29 +-3.2     |
| OEWC    |   62.18% +- 0.7  |    32.33 +-3.4     |
| EWC     |   63.92% +- 6.6  |    31.91 +-3.2     |
| HM(ours)|   69.36% +- 6.6  |    22.64 +-2.8     |
| Multi   |   86.09% +- 0.1  |        N/A         |
| Indep   |   92.24% +- 0.7  |        N/A         |



### References

[1] Kirkpatrick, James, et al. "Overcoming catastrophic forgetting in neural networks." Proceedings of the national academy of sciences 114.13 (2017): 3521-3526.

[2] Schwarz, Jonathan, et al. "Progress & compress: A scalable framework for continual learning." arXiv preprint arXiv:1805.06370 (2018).

[3] Lee, Sang-Woo, et al. "Overcoming catastrophic forgetting by incremental moment matching." Advances in neural information processing systems. 2017.
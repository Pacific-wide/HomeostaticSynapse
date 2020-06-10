

# Title

This repository is the official implementation of Homeostasis-Inspired Continual Learning: Learning to Control Structural Regularization


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

via Anaconda

## Training

To train the Homeostatic Meta-Model in the paper, run this command:

```train
python meta_alpha_train.py --data MNISTBPERM --n_task 30 --seed 0
```


## Evaluation

To evaluate Homeostatic Meta-Model on MNIST-BPERM, run:

```eval
python meta_alpha_test.py --data MNISTPERM
```

> ??Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Alternativv Models

You can evaluate alternative methods


- Single : a single learner based on SGD for a sequence of tasks
- InDep : a dedicated (independent) learner based on SGD for each task
- EWC : Elastic Weight Consolidation (regularized with the dedicated Fisher information for each task) [1]
- OEWC : Online Elastic Weight Consolidation (regularized with the accumulated Fisher information) [2]
- IMM  : Incremental Moment Matching with a weight transfer method [3]
- Multi-task : allowed to access all the tasks (violation of strict CL scenario).

## Results

Our model achieves the following performance on :

### MNIST-PERM Dataset

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |


### References

[1] Kirkpatrick, James, et al. "Overcoming catastrophic forgetting in neural networks." Proceedings of the national academy of sciences 114.13 (2017): 3521-3526.

[2] Schwarz, Jonathan, et al. "Progress & compress: A scalable framework for continual learning." arXiv preprint arXiv:1805.06370 (2018).

[3] Lee, Sang-Woo, et al. "Overcoming catastrophic forgetting by incremental moment matching." Advances in neural information processing systems. 2017.
# My Paper Title

This repository is the official implementation of Model.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> Recommend to use latest version of anaconda. 

## Meta-Training

To train the model(s) in the paper, run this command:

```train
python meta_alpha_train.py --input-data <path_to_data> --alpha 10 --beta 20
```

> Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Meta-Testing (Evaluation)

To evaluate my model on MNIST-PERM, run:

```eval
python meta_alpha_test.py --model-file mymodel.pth --benchmark imagenet
```

> Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

> ??Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

> Pick a licence and describe how to contribute to your code repository. 

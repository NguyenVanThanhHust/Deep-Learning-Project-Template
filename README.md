# PyTorch Project Template

# Requirements
- [yacs](https://github.com/rbgirshick/yacs) (Yet Another Configuration System)
- [PyTorch](https://pytorch.org/) (An open source deep learning platform) 
- [ignite](https://github.com/pytorch/ignite) (High-level library to help with training neural networks in PyTorch)

# Table Of Contents
-  [How to run](#how-to-run)
-  [Acknowledgments](#acknowledgments)

# How to run   
How to train
```
python tools/train.py --config_file configs/train_mnist_softmax.yaml
```

How to infer
```
python tools/test --config_file configs/eval_mnist.yaml
```

# Acknowledgments

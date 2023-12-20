# Crack Segmentation

# Table Of Contents
-  [How to run](#how-to-run)
-  [Acknowledgments](#acknowledgments)

# How to run
Download data from [crack_segmentation_repo](https://github.com/khanhha/crack_segmentation)
Split train set to train and validation set, modify `data_folder` accord to your dataset folder
```
python scripts/split_train_val.py --data_folder ../Datasets/ 
```   

How to train
```
python tools/train.py --config_file configs/simple_unet.yaml PRETRAINED_CHECKPOINT if_you_have_pretrained_ckpt
```

How to infer
```
python tools/test.py --config_file configs/simple_unet.yaml TEST.WEIGHT your_trained_weight_here
```

How to visualize prediction
```
python tools/viz_prediction.py --config_file configs/simple_unet.yaml --input_folder your_image_folder TEST.WEIGHT your_trained_weight_here
```

# Acknowledgments
Unet Implementation: https://github.com/usuyama/pytorch-unet
Deep-Learning project template: 
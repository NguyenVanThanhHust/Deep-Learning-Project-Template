# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import torch

from .unet import UNet
from .losses import Loss

def build_model(cfg):
    model = UNet(cfg.MODEL.NUM_CLASSES)
    if cfg.PRETRAINED_CHECKPOINT is not None:
        ckpt = torch.load(cfg.PRETRAINED_CHECKPOINT)
        model.load_state_dict(ckpt["model_state_dict"])
        print("Load model state dict from: ", cfg.PRETRAINED_CHECKPOINT)
    return model

def build_losses(cfg):
    return Loss()
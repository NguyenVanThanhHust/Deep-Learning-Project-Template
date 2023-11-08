# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import math
import os
import sys
from typing import Iterable

import torch

@torch.no_grad()
def evaluate(model, data_loader, metric, device, logger):
    model = model.to(device)
    metric = metric.to(device)
    model.eval()
    
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            accc = metric(outputs, targets)
        
    acc = metric.compute() 
    logger.info("Acc {}".format(acc))
    return acc
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import math
import os
from os.path import join
import sys
from typing import Iterable

import torch


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    scheduler, 
                    device: torch.device, epoch: int, logger, writer):
    model.train()
    model = model.to(device)

    for idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        losses = criterion(outputs, targets)
        loss_value = losses.item()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        logger.info("loss: {}".format(loss_value))
        writer.add_scalar("Loss/train", loss_value, epoch)

@torch.no_grad()
def evaluate(model, criterion, data_loader, metric, device, epoch, logger, writer):
    model.eval()
    model = model.to(device)
    metric = metric.to(device)

    for idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        losses = criterion(outputs, targets)
        acc = metric(outputs, targets)
        loss_value = losses.item()
        logger.info("loss: {}".format(loss_value))
        writer.add_scalar("Loss/eval", loss_value)
        writer.add_scalar("acc", acc)

    acc = metric.compute() 
    logger.info("Epoch: {} acc {}".format(epoch, acc))
    return acc


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        device, 
        optimizer,
        scheduler,
        loss_fn,
        metric, 
        logger, 
        writer, 
        output_dir, 
):
    epochs = cfg.SOLVER.MAX_EPOCHS
    logger = logging.getLogger("template_model.train")
    logger.info("Start training")
    best_acc = 0.0

    for epoch in range(epochs):
        train_one_epoch(model, loss_fn, train_loader, optimizer, scheduler, device, epoch, logger, writer)
        acc = evaluate(model, loss_fn, val_loader, metric, device, epoch, logger, writer)
        if acc > best_acc:
            best_acc = acc
            best_model_path = join(output_dir, 'best_ckpt.pth')
            logger.info("Save best model at epoch: {}".format(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, best_model_path)
        latest_model_path = join(output_dir, 'latest_ckpt.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, latest_model_path)
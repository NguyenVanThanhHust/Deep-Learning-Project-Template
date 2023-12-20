# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir
from os.path import join, isfile, isdir
from tqdm import tqdm 

import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch.nn.functional as F

sys.path.append('.')
from config import cfg
from modeling import build_model
from utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="PyTorch Template MNIST Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "--input_folder", default="", help="path to image folder", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)
    
   
    
    logger = setup_logger("simple_unet", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    device = torch.device(cfg.MODEL.DEVICE)

    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT)["model_state_dict"])

     # Create output visalize folder
    output_path = cfg.TEST.WEIGHT
    folders = output_path.split("/")
    vis_folder = join(folders[0], folders[1], "vis")
    os.makedirs(vis_folder, exist_ok=True)


    input_folder = args.input_folder
    image_files = next(os.walk(input_folder))[2]

    model.eval()
    model = model.to(device)
    print("Device: ", device)
    mean=cfg.INPUT.PIXEL_MEAN
    std=cfg.INPUT.PIXEL_STD

    for im_file in tqdm(image_files):
        im_path = join(input_folder, im_file)
        ori_im = cv2.imread(im_path)
        im = cv2.resize(ori_im, cfg.INPUT.SIZE_TEST)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(np.float32)/255.0
        for i, (m, s) in enumerate(zip(mean, std)):
            im[:, :, i] -= m
            im[:, :, i] /= s
        inputTensor = transforms.ToTensor()(im)
        inputTensor = inputTensor.to(device)
        inputTensor = torch.unsqueeze(inputTensor, dim=0)
        output = model(inputTensor)
        output = F.sigmoid(output)
        output_cpu = output.detach().cpu().numpy()
        output_cpu = output_cpu * 255.0
        output_cpu = output_cpu.astype(int)
        output_img_path = join(vis_folder, im_file)
        output_cpu = output_cpu[0, 0, :, :]
        output_cpu = np.stack((output_cpu, output_cpu, output_cpu), axis=0)
        output_cpu = np.moveaxis(output_cpu, 0, -1)
        output_cpu = cv2.resize(output_cpu, (ori_im.shape[0], output_cpu.shape[1]))
        vis_img = cv2.hconcat((ori_im, output_cpu.astype(ori_im.dtype)))
        cv2.imwrite(output_img_path, vis_img)

if __name__ == '__main__':
    main()

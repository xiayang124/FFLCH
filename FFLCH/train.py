import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.optim as optim
import os
import argparse

from segment_anything import sam_model_registry, SamPredictor
from Simple_Net import Simple_Net
from data_process import read_mat
from set_args import get_arg


def train_process(HSI, Label, param):
    max_classes = Label.max()
    param.max_classes = max_classes



if __name__ == "__main__":
    params = argparse.ArgumentParser(description="Training Process Params")
    params.add_argument("--hsi", type=str, default="./dataset/hsi/Pavia.mat", help="HSI dir")
    params.add_argument("--label", type=str, default="./dataset/label/Pavia_gt.mat", help="Lable dir")
    params.add_argument("--device", type=str, default="gpu0", help="choose device for training")
    params.add_argument("--checkpoint_path", type=str, default="../segment_anything/sam_vit_b_01ec64.pth",
                        help="sam model path, must be same as model")
    params.add_argument("--mode", type=str, default="vit_b", help="sam model, must be same as checkpoint_path")
    params.add_argument("--batch_size", type=int, default=1, help="batch_size")
    params.add_argument("--learning_rate", type=float, default=0.005, help="learning rate")
    params.add_argument("--train_rate", type=float, default=0.7, help="the rate of total train data")
    params.add_argument("--train_point", type=int, default=10, help="the num of one epoch num")
    params.add_argument("--max_classes", type=int, default=0)
    params.add_argument("--input_sam", type=float, default=0.3, help="the rate of train point to sam")
    params.add_argument("--epoch", type=int, default=100, help="epoch")
    params.add_argument("--if_sam", type=bool, default=True, help="if use sam")
    params.add_argument("--if_mlp", type=bool, default=True, help="if use mlp")
    arg = params.parse_args()

    HSI_name = "Pavia"

    args = get_arg(HSI_name, arg)

    HSI = read_mat(args.hsi)
    Label = read_mat(args.label)

    train_process(HSI, Label, args)

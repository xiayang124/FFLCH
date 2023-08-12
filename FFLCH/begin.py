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
from data_process import data_aug, set_band
from get_argparse import set_args


def train_process(HSI, Label, param):
    aug = data_aug(HSI, Label, param.max_classes, param.train_num)
    # Get train location and test location
    train_location, test_location = aug.segment_data()
    # Set test data args
    aug.test_data_num()
    # Seg to 3 band data
    augment_HSI = set_band(HSI, param.train_num, mode="random")


if __name__ == "__main__":
    HSI_name = "Pavia"

    # Get the args and data
    arg = set_args(HSI_name)
    args, HSI, Label = arg.get_arg()

    train_process(HSI, Label, args)

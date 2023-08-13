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
from data_process import data_aug, set_pca
from get_argparse import set_args
from train import model, train


def train_process(HSI, Label, param):
    aug = data_aug(HSI, Label, param.max_classes, param.train_num)
    # Get train location and test location
    train_location, test_location = aug.segment_data()
    # Set test data args
    aug.set_test_data_num()
    # Seg to 3 band data
    pca_HSI = set_pca(HSI, choose_band=3)

    trains = train(in_channel=3, out_channel=3, train_location=train_location, test_location=test_location, param=param)
    trains.train_mode(True, False)

    trains.train_process(pca_HSI, Label)


if __name__ == "__main__":
    HSI_name = "Pavia"

    # Get the args and data
    arg = set_args(HSI_name)
    args, HSI, Label = arg.get_arg()

    train_process(HSI, Label, args)

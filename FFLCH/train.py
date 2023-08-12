import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from segment_anything import sam_model_registry, SamPredictor
from Simple_Net import Simple_Net


class model(nn.Module):
    """
    A container of model param
    """
    def __init__(self, in_channal: int, out_channal: int, param):
        super().__init__()
        self.loss = nn.CrossEntropyLoss().to(param.device)
        self.MLP = Simple_Net(in_channal, out_channal).to(param.device)
        self.SAM = sam_model_registry[param.mode](param.checkpoint_path).to(param.device)
        self.Adam = optim.Adam(self.MLP.parameters(), lr=param.learning_rate)
        self.mlp_train = True
        self.sam_train = True

    def train_mode(self, mlp_train=True, sam_train=False):
        """
        Choose the net which will be frozen. If sam_train is True, the network will fine-tune,
        or the sam will be frozen. mlp_train and sam_train all False will cause the net never
        upgrade.

        :param mlp_train: Whether freeze the MLP block, defalut True
        :param sam_train: Whether freeze the SAM block, defalut False
        """
        if not mlp_train:
            for param in self.MLP.parameters():
                param.requires_grad = False
        if sam_train:
            for param in self.SAM.parameters():
                param.requires_grad = True
        else:
            for param in self.SAM.parameters():
                param.requires_grad = False

    def feed_net(self, mlp_train=True, sam_train=True):
        """
        Choose the net which is needed. MUST one of them is True at least.
        :param mlp_train:
        :param sam_train:
        :return:
        """
        self.mlp_train = mlp_train
        self.sam_train = sam_train


class train(nn.Module):
    def __init__(self, model: model, ):
        super().__init__()
        self.model = model

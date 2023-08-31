import torch
import torch.nn as nn
import torch.optim as optim
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from FFLCH.simple_net import Simple_Net


class model(nn.Module):
    """
    Model settings.
    """
    def __init__(self, in_channal: int, out_channal: int, param):
        super().__init__()
        self.param = param
        self.in_channal = in_channal
        self.out_channal = out_channal
        self.device = param.device
        self.loss = nn.BCEWithLogitsLoss().to(self.device)
        self.IntoSAM = Simple_Net(in_channal, out_channal, if_norm=False, if_hide_layer=True).to(self.device)
        # self.AfterSAM = Simple_Net(1, 1, if_norm=False, if_hide_layer=True).to(self.device)
        self.SAM = sam_model_registry[param.mode](param.checkpoint_path).to(self.device)
        self.Adam = optim.Adam(self.IntoSAM.parameters(), lr=param.learning_rate)
        self.sam = SamPredictor(self.SAM)
        self.PegSam = SamAutomaticMaskGenerator(self.SAM)

        self.if_mlp_need = True
        self.if_sam_need = True

    def train_mode(self, mlp_train=True, sam_train=False):
        """
        Choose the net which want to be frozen. If sam_train is True, the network will fine-tune,
        otherwise the sam will be frozen. mlp_train and sam_train all False will cause the net never
        backward.

        :param mlp_train: Whether freeze the IntoSAM block, defalut True
        :param sam_train: Whether freeze the SAM block, defalut False
        """
        self.mlp_train = mlp_train
        self.sam_train = sam_train
        if not mlp_train:
            for param in self.IntoSAM.parameters():
                param.requires_grad = False
        if sam_train:
            for param in self.SAM.parameters():
                param.requires_grad = True
        else:
            for param in self.SAM.parameters():
                param.requires_grad = False

    def feed_net(self, mlp_train=True, sam_train=True):
        """
        Choose the net which is needed. MUST one of them is used(Ture) at least.
        :param mlp_train: Whether the mlp layer is needed, defalut True
        :param sam_train: Whether the sam layer is needed, defalut True
        """
        assert mlp_train or sam_train, f"Please choose a net to use at least!"
        self.if_mlp_need = mlp_train
        self.if_sam_need = sam_train
        # If mlp is not needed, set 1*1 conv to blank
        if not mlp_train:
            self.param.epochs = 1
            self.MLP = nn.Identity()

    def set_model_params(self, optim: optim.Optimizer, loss):
        self.Adam = optim
        self.loss = loss

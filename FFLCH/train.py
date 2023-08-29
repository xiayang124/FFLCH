import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

import data_process
import utils.eval as evals
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from simple_net import Simple_Net


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
        self.MLP = Simple_Net(in_channal, out_channal, if_norm=False).to(self.device)
        self.SAM = sam_model_registry[param.mode](param.checkpoint_path).to(self.device)
        self.Adam = optim.Adam(self.MLP.parameters(), lr=param.learning_rate)
        self.sam = SamPredictor(self.SAM)
        self.PegSam = SamAutomaticMaskGenerator(self.SAM)

        self.if_mlp_need = True
        self.if_sam_need = True

    def train_mode(self, mlp_train=True, sam_train=False):
        """
        Choose the net which will be frozen. If sam_train is True, the network will fine-tune
        or the sam will be frozen. mlp_train and sam_train all False will cause the net never
        backward.

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
        :param mlp_train: Whether the mlp layer is needed, defalut True
        :param sam_train: Whether the sam layer is needed, defalut True
        """
        assert mlp_train or sam_train, f"Please choose a net to use at least!"
        self.if_mlp_need = mlp_train
        self.if_sam_need = sam_train
        if not mlp_train:
            self.param.epochs = 1

    def set_model_params(self, optim: optim.Optimizer, loss):
        self.Adam = optim
        self.loss = loss


class train(model):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            param: argparse.Namespace,
            input_label: np.ndarray,
            loss_label: torch.Tensor,
            train_label: torch.Tensor,
            test_label: np.ndarray,
            input_location: np.ndarray,
            loss_location: torch.Tensor,
            train_location: np.ndarray,
            test_location: np.ndarray,
            current_class: int
    ):
        """
        :param in_channel: MLP in channel
        :param out_channel: MLP out channel
        :param param: Args
        :param input_label: The label which can distinguish background and foreground input SAM.
        :param loss_label: The label used in loss
        :param train_label: The test label
        :param input_location: The label location of input sam
        :param loss_location: The label location of loss
        :param train_location: The test label location of input sam
        :param test_location: The test label location of loss
        :param current_class: Current training class
        """
        super().__init__(in_channal=in_channel, out_channal=out_channel, param=param)
        self.input_label = input_label
        self.loss_label = loss_label
        self.train_label = train_label
        self.test_label = test_label

        self.input_location = input_location
        self.loss_location = loss_location
        self.train_location = train_location
        self.test_location = test_location

        self.current_class = current_class

    def train_process(self, HSI: np.ndarray, Label: np.ndarray):
        """
        Train process.
        :param HSI: Processed Data(PCA)
        :param Label: Initial label
        """
        # For each class
        assert self.if_sam_need or self.if_mlp_need, f"Wrong set of feed net!"
        # ------------------------------------------------Begin Training------------------------------------------------
        # If mlp and sam is all needed
        if self.if_sam_need:
            # Show the pic of pca -> 1*1 conv (testing)
            # data_process.show_pic(HSI, input_location, self.per_class, self.input_num, "PCA")
            # Sample data process
            torch_train_pic = torch.from_numpy(HSI.astype("int32")).to(self.device)
            # [h, w, b] -> [b, h, w] -> [1, b, h, w]
            torch_train_pic = torch.unsqueeze(torch_train_pic.permute((2, 0, 1)), dim=0)
            for epoch in range(self.param.epochs):
                self.MLP.train()
                begin_time = time.time()
                # Feed net
                losses, OA = self._mlp_sam_forward(HSI.shape,
                                                   torch_train_pic,
                                                   self.input_location,
                                                   self.input_label,
                                                   self.loss_location,
                                                   self.loss_label,
                                                   epoch,
                                                   self.param,
                                                   if_mlp=self.if_mlp_need)
                if self.if_mlp_need:
                    self.Adam.zero_grad()
                    losses.backward()
                    self.Adam.step()
                end_time = time.time()
                print(f"epoch {epoch}, loss is {losses}, use {end_time - begin_time} s, acc is {OA}")
                # ---------------------------------------------Train End------------------------------------------------
                # -----------------------------------------------Test---------------------------------------------------
                if epoch % 20 == 0:
                    self.MLP.eval()
                    losses, OAs = self._mlp_sam_forward(HSI.shape,
                                                        torch_train_pic,
                                                        self.train_location,
                                                        self.train_label,
                                                        self.test_location,
                                                        self.test_label,
                                                        epoch,
                                                        self.param,
                                                        if_test=True,
                                                        if_mlp=self.if_mlp_need)
                    print(f"test OA is {OAs}, epoch is {epoch}")
                # ------------------------------------------Test End-----------------------------------------------
                if epoch == self.param.epochs - 1:
                    data_process.write_file(self.current_class, OAs)
                    torch.save(self.MLP.state_dict(), "./pth/class/" + str(self.current_class))

    def _mlp_sam_forward(
            self,
            shape,
            data: torch.Tensor,
            input_location: np.ndarray,
            input_label: np.ndarray,
            data_location: np.ndarray,
            loss_label: torch.Tensor,
            epoch: int,
            param: argparse.Namespace,
            if_test=False,
            if_mlp=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward stage.
        :param shape: Initial HSI shape
        :param data: The output of SAM data argument
        :param input_location: The location about input SAM
        :param input_label: The label which distinguish the fore and background about input SAM
        :param data_location: The location of data
        :param loss_label: The label which distinguish the fore and background about loss
        :param if_test: Whether process with test mode. Defalut False(Train mode)
        :return: losses, OA
        """
        height, width, band = shape
        location_num, _ = data_location.shape
        inputs = data.clone()
        # Feed into MLP
        if if_mlp:
            mlp_out = self.MLP.forward(inputs.float())
            # MLP out as the input of sam augment
            sam_aug = self.sam.get_apply_image(mlp_out)
        else:
            sam_aug = self.sam.get_apply_image(inputs)
        self.sam.set_torch_image(sam_aug, (height, width))
        # Predict
        sam_out, _, _ = self.sam.predict(input_location, input_label, return_logits=not if_test, multimask_output=False)
        data_location_x, data_location_y = torch.chunk(data_location, 2, dim=1)
        # Get target location mask
        sam_mask_out = sam_out[0, data_location_y.long(), data_location_x.long()]
        sam_mask_out = torch.squeeze(sam_mask_out)
        # Train mode
        if not if_test:
            losses = self.loss(sam_mask_out, loss_label.float())
            OA = evals.OA(sam_mask_out, loss_label, if_replace01=True)
        # Test mode
        else:
            sam_mask_out = sam_mask_out.int()
            # sam aug -> 1*1 conv
            sam_aug = sam_aug.detach().cpu().numpy()
            sam_aug = np.squeeze(sam_aug, axis=0)
            data_process.show_pic(sam_aug, input_location, self.current_class, param.train_num, epoch, "Argument")
            # OA
            OA = evals.OA(sam_mask_out, loss_label, if_replace01=False)
            # sam output -> final
            data_process.show_pic(sam_out, input_location, self.current_class, param.train_num, epoch, "test", replace=if_test)
            losses = 0
        # OA
        return losses, OA

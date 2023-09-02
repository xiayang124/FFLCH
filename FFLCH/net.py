import argparse
from typing import Tuple

import torch
import numpy as np
import time

import data_process
from utils.eval import per_acc as aa
from utils.model import model as model


class FFLCHs(model):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            param: argparse.Namespace,
            input_location: np.ndarray,
            loss_location: torch.Tensor,
            train_location: np.ndarray,
            test_location: np.ndarray
    ):
        """
        :param in_channel: IntoSAM in channel
        :param out_channel: IntoSAM out channel
        :param param: Args
        :param input_location: The label location of input sam
        :param loss_location: The label location of loss
        :param train_location: The test label location of input sam
        :param test_location: The test label location of loss
        """
        super().__init__(in_channal=in_channel, out_channal=out_channel, param=param)

        self.input_location = input_location
        self.loss_location = loss_location
        self.train_location = train_location
        self.test_location = test_location

        self.input_label = None
        self.loss_label = None
        self.train_label = None
        self.test_label = None
        self.current_class = None

    def set_label(
            self,
            input_label: np.ndarray,
            loss_label: torch.Tensor,
            train_label: torch.Tensor,
            test_label: np.ndarray,
            current_class: int,
            epoch: int
    ):
        self.input_label = input_label
        self.loss_label = loss_label
        self.train_label = train_label
        self.test_label = test_label

        self.current_class = current_class
        self.epoch = epoch

    def per_class_training(self, HSI: np.ndarray, Label: np.ndarray):
        """
        Train process.
        :param HSI: Processed Data(PCA)
        :param Label: Initial label
        """
        height, width, band = HSI.shape
        # Show the pic of pca -> 1*1 conv (testing)
        # data_process.show_pic(HSI, input_location, self.per_class, self.input_num, "PCA")
        torch_train_pic = torch.from_numpy(HSI.astype("int32")).to(self.device)  # Sample data process
        torch_train_pic = torch.unsqueeze(torch_train_pic.permute((2, 0, 1)), dim=0)  # [h, w, b] -> [b, h, w] -> [1, b, h, w]
        # ------------------------------------------------Begin Training------------------------------------------------
        begin_time = time.time()
        self.IntoSAM.train()
        losses, per_acc = self.train_process(HSI.shape, torch_train_pic)
        end_time = time.time()
        print(f"epoch {self.epoch}, loss is {float(losses)}, use {end_time - begin_time} s, acc is {per_acc}")
        # ---------------------------------------------------Train End--------------------------------------------------
        # -----------------------------------------------------Test-----------------------------------------------------
        if self.epoch % 25 == 0 or self.epoch == 1:
            self.IntoSAM.eval()
            per_accs, _, mlp_out, sam_out = self.predict(HSI.shape, torch_train_pic)
            # Test use only
            data_process.show_pic(mlp_out, self.train_location, self.current_class, self.param.train_num, self.epoch, "MLP")
            data_process.show_pic(sam_out, self.train_location, self.current_class, self.param.train_num, self.epoch,
                                  "test", replace=True)
            print(f"test AA is {per_accs}, epoch is {self.epoch}")
        # ----------------------------------------------------Test End--------------------------------------------------
        if self.epoch == self.param.epochs and self.epoch != 1:
            torch.save(self.IntoSAM.state_dict(), "./pth/class/IntoSAM" + str(self.current_class))

    @torch.no_grad()
    def predict(self, shape, HSI: torch.Tensor):
        # Deep Copy
        input_conv = HSI.clone()
        height, width, band = shape
        # 2 1*1 Conv or blank(None)
        mlp_out = self.IntoSAM(input_conv.float())
        # SAM's data process
        sam_aug = self.sam.get_apply_image(mlp_out)
        # Set image
        self.sam.set_torch_image(sam_aug, (height, width))
        # Input SAM
        sam_out, _, _ = self.sam.predict(self.train_location, self.train_label, return_logits=False)
        # Split axis (Attention: Because of different coordinate system, axis x and axis y had been exchanged)
        data_location_x, data_location_y = torch.chunk(self.test_location, 2, dim=1)
        sam_mask_out = sam_out[0, data_location_y.long(), data_location_x.long()]
        sam_mask_out = torch.squeeze(sam_mask_out).int()
        mlp_out = mlp_out.detach().cpu().numpy()
        # Precise
        per_acc, same = aa(sam_mask_out, self.test_label, if_replace01=False, if_final=True)
        return per_acc, same, mlp_out, sam_out

    def train_process(self, shape: Tuple, HSI: torch.Tensor):
        input_conv = HSI.clone()
        height, width, band = shape
        # 2 1*1 conv or blank(None)
        mlp_out = self.IntoSAM(input_conv.float())
        # SAM's data process
        sam_aug = self.sam.get_apply_image(mlp_out)
        # Set image
        self.sam.set_torch_image(sam_aug, (height, width))
        # Input SAM
        sam_out, _, _ = self.sam.predict(self.input_location, self.input_label, return_logits=True)
        # Split axis (Attention: Because of different coordinate system, axis x and axis y had been exchanged)
        data_location_x, data_location_y = torch.chunk(self.loss_location, 2, dim=1)
        sam_mask_out = sam_out[0, data_location_y.long(), data_location_x.long()]
        sam_mask_out = torch.squeeze(sam_mask_out)

        losses = self.loss(sam_mask_out, self.loss_label.float())
        # AA
        per_acc = aa(sam_mask_out, self.loss_label, if_replace01=True)

        if self.if_mlp_need and self.mlp_train:
            self.Adam.zero_grad()
            losses.backward()
            self.Adam.step()
        return losses, per_acc

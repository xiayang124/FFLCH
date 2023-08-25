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
from Simple_Net import Simple_Net


class model(nn.Module):
    """
    Model settings.
    """
    def __init__(self, in_channal: int, out_channal: int, param):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss().to(param.device)
        self.MLP = Simple_Net(in_channal, out_channal, if_norm=False).to(param.device)
        self.SAM = sam_model_registry[param.mode](param.checkpoint_path).to(param.device)
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

    def set_model_params(self, optim: optim.Optimizer, loss):
        self.Adam = optim
        self.loss = loss

class train(model):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            param
    ):
        super().__init__(in_channal=in_channel, out_channal=out_channel, param=param)

    def train_process(self, HSI: np.ndarray, Label: np.ndarray, train_location: dict, test_location: dict, param):
        """
        Train process.
        :param param: Dataset param
        :param test_location: Test location data
        :param train_location: Train location data
        :param HSI: Processed Data(PCA)
        :param Label: Initial label
        """
        # ------------------------------------------------Data Process--------------------------------------------------
        # Initial data process class
        data_aug = data_process.data_argument(HSI, Label, train_location, test_location, param)
        # The num of input and Loss
        self.input_num, self.label_num = data_aug.get_input_num()
        # Get test data per class min num
        test_min_num, _ = data_aug.test_num()
        # Get the input sam location, loss location and test location
        input_location, label_location = data_aug.split_location()
        test_input_location, _ = data_aug.split_location(input_num=test_min_num, if_test=True)
        # Get the input sam label, loss label and test label
        all_input_label, all_loss_label = data_aug.generate_sam_label()
        test_label, _ = data_aug.generate_sam_label(test_min_num, if_test=True)

        # For each class
        for self.per_class in range(param.max_classes):
            input_label, loss_label = all_input_label[self.per_class], all_loss_label[self.per_class]
            # Get SAM output index mask and test mask
            train_sam_mask = data_aug.mask_generate(label_location)
            test_mask = data_aug.mask_generate(test_input_location)

            loss_label = torch.from_numpy(loss_label).to(param.device)
            train_sam_mask = torch.from_numpy(train_sam_mask).to(param.device)
            test_mask = torch.from_numpy(test_mask).to(param.device)
            test_labels = torch.from_numpy(test_label[self.per_class]).to(param.device)

            assert self.if_sam_need or self.if_mlp_need, f"Wrong set of feed net!"
            # --------------------------------------------------End-----------------------------------------------------
            # ---------------------------------------------Begin Training-----------------------------------------------
            # If mlp and sam is all needed
            if self.if_sam_need:
                # Show the pic of pca -> 1*1 conv (testing)
                data_process.show_pic(HSI, input_location, self.per_class, self.input_num, "PCA")
                # Sample data process
                torch_train_pic = torch.from_numpy(HSI.astype("int32")).to(param.device)
                # [h, w, b] -> [b, h, w] -> [1, b, h, w]
                torch_train_pic = torch.unsqueeze(torch_train_pic.permute((2, 0, 1)), dim=0)

                for epoch in range(param.epochs):
                    self.MLP.train()
                    begin_time = time.time()
                    # Feed net
                    losses, OA = self._mlp_sam_forward(
                        HSI.shape,
                        torch_train_pic,
                        input_location,
                        input_label,
                        train_sam_mask,
                        loss_label,
                        param,
                        if_mlp=self.if_mlp_need
                    )
                    if self.if_mlp_need:
                        self.Adam.zero_grad()
                        losses.backward()
                        self.Adam.step()

                    end_time = time.time()
                    print(f"epoch {epoch}, loss is {losses}, use {end_time - begin_time} s, acc is {OA}")
                    # -------------------------------------------Train End----------------------------------------------
                    # ---------------------------------------------Test-------------------------------------------------
                    if epoch % 20 == 0:
                        self.MLP.eval()
                        losses, OAs = self._mlp_sam_forward(
                            HSI.shape,
                            torch_train_pic,
                            np.concatenate((input_location, label_location), axis=0),
                            np.concatenate((input_label, loss_label.clone().cpu()), axis=0),
                            test_mask,
                            test_labels,
                            param,
                            if_test=True,
                            if_mlp=self.if_mlp_need
                        )
                        print(f"test OA is {OAs}, epoch is {epoch}")
                    # ------------------------------------------Test End-----------------------------------------------
                    if epoch == param.epochs - 1:
                        torch.save(self.MLP.state_dict(), "./pth/class" + str(self.per_class))

    def _mlp_sam_forward(
            self,
            shape,
            data: torch.Tensor,
            input_location: np.ndarray,
            input_label: np.ndarray,
            train_sam_mask: np.ndarray,
            loss_label: torch.Tensor,
            param: argparse.Namespace,
            if_test=False,
            if_mlp=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward of MLP and SAM.
        :param shape: Initial HSI shape
        :param data: The output of SAM data argument
        :param input_location: The location about input SAM
        :param input_label: The label which distinguish the fore and background about input SAM
        :param train_sam_mask: the mask of sam output
        :param loss_label: The label which distinguish the fore and background about loss
        :param if_test: Whether process with test mode. Defalut False(Train mode)
        :return: losses, OA
        """
        height, width, band = shape
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
        # Get target location mask
        sam_mask_out = sam_out[0, train_sam_mask]
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
            data_process.show_pic(sam_aug, input_location, self.per_class, param.train_num, "Argument")
            # OA
            OA = evals.OA(sam_mask_out, loss_label, if_replace01=False)
            # sam output -> final
            data_process.show_pic(sam_out, input_location, self.per_class, param.train_num, None, replace=if_test,
                                  save=if_test)
            losses = 0
        # OA
        return losses, OA

    @torch.no_grad()
    def perdict(self):
        pass

    def all_generate(self, HSI):
        pass

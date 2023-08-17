from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

import data_process
import utils.eval as evals
from segment_anything import sam_model_registry, SamPredictor
from Simple_Net import Simple_Net


class model(nn.Module):
    """
    A container of model param
    """

    def __init__(self, in_channal: int, out_channal: int, param):
        super().__init__()
        self.loss = nn.BCELoss().to(param.device)
        self.MLP = Simple_Net(in_channal, out_channal, if_norm=False).to(param.device)
        self.SAM = sam_model_registry[param.mode](param.checkpoint_path).to(param.device)
        self.Adam = optim.Adam(self.MLP.parameters(), lr=param.learning_rate)
        self.if_mlp_need = param.if_mlp
        self.if_sam_need = param.if_sam

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
        :param mlp_train: Whether the mlp layer is needed,
        :param sam_train: Whether the sam layer is needed
        """
        assert mlp_train or sam_train, f"Please choose a net to use at least!"
        self.if_mlp_need = mlp_train
        self.if_sam_need = sam_train

    # TODO(Byan Xia): Add expression
    def set_model_params(self):
        pass


# TODO(Byan Xia): Add evaluation module and contract module
class train(model):
    """
    Training process
    """
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            train_location: dict,
            test_location: dict,
            param
    ):
        super().__init__(in_channal=in_channel, out_channal=out_channel, param=param)
        self.train_location = train_location
        self.test_location = test_location
        self.sam = SamPredictor(self.SAM)
        self.param = param

    def train_process(self, HSI: np.ndarray, Label: np.ndarray):
        """
        Train process
        :param HSI: Processed Data(PCA)
        :param Label: Initial label
        """
        # The num of input and Loss
        self.input_num, self.label_num = data_process.get_input_num(self.param.train_num, self.param.input_sam)

        # For each class
        for self.per_class in range(self.param.max_classes):
            # ----------------------------------------------Data Process------------------------------------------------
            # The num of min test data
            test_min_num = int(data_process.test_min_data(self.test_location))
            # Get the input sam location, loss location and test location
            input_location, label_location = data_process.split_location(self.train_location, self.input_num,
                                                                         self.label_num, self.per_class)
            test_input_location, _ = data_process.split_location(self.test_location, test_min_num, 0, self.per_class)
            # Get the input sam label, loss label and test label
            input_label, loss_label = data_process.generate_sam_label(self.input_num, self.label_num, self.per_class,
                                                                      self.param.max_classes)
            test_input_label, _ = data_process.generate_sam_label(test_min_num, 0, self.per_class, self.param.max_classes)
            # Get SAM output index mask and test mask
            train_sam_mask = data_process.mask_generate(label_location, Label)
            test_mask = data_process.mask_generate(test_input_location, Label)

            # Shape
            height, width, band = HSI.shape
            loss_label = torch.from_numpy(loss_label).to(self.param.device)
            train_sam_mask = torch.from_numpy(train_sam_mask).to(self.param.device)
            test_mask = torch.from_numpy(test_mask).to(self.param.device)
            test_input_label = torch.from_numpy(test_input_label).to(self.param.device)

            assert self.if_sam_need or self.if_mlp_need, f"Wrong set of feed net!"
            # --------------------------------------------------End-----------------------------------------------------
            # ---------------------------------------------Begin Training-----------------------------------------------
            # If mlp and sam is all needed
            if self.if_sam_need:
                # Testing
                classes = 0
                # Show the pic of pca -> 1*1 conv (testing)
                data_process.show_pic(HSI, input_location, self.per_class, None, "PCA")
                # Sample data process
                torch_train_pic = torch.from_numpy(HSI.astype("int32")).to(self.param.device)
                # [h, w, b] -> [b, h, w] -> [1, b, h, w]
                torch_train_pic = torch.unsqueeze(torch_train_pic.permute((2, 0, 1)), dim=0)

                for epoch in range(self.param.epochs):
                    begin_time = time.time()
                    # Feed net
                    losses, OA = self._mlp_sam_forward(HSI.shape,
                                                       torch_train_pic,
                                                       input_location,
                                                       input_label,
                                                       train_sam_mask,
                                                       loss_label,
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
                        torch_test_pic = torch.clone(torch_train_pic)

                        losses, OAs = self._mlp_sam_forward(HSI.shape,
                                                            torch_test_pic,
                                                            input_location,
                                                            input_label,
                                                            test_mask,
                                                            test_input_label,
                                                            if_test=True,
                                                            if_mlp=False
                                                            )
                        print(f"test OA is {OAs}, epoch is {epoch}")
                    # ------------------------------------------Test End-----------------------------------------------
                classes += 1

    # TODO(Byan Xia): Add expression
    def _mlp_sam_forward(
            self,
            shape,
            data: torch.Tensor,
            input_location: np.ndarray,
            input_label: np.ndarray,
            train_sam_mask: np.ndarray,
            loss_label: torch.Tensor,
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
        :return: losses and OA
        """
        height, width, band = shape
        data = (data - torch.min(data)) / (torch.max(data) - torch.min(data)) * 255
        # Feed into MLP
        if if_mlp:
            mlp_out = self.MLP.forward(data.float())
            # MLP out as the input of sam augment
            sam_aug = self.sam.get_apply_image(mlp_out)
        else:
            sam_aug = self.sam.get_apply_image(data)
        self.sam.set_torch_image(sam_aug, (height, width))
        # Predict
        sam_out, score, _ = self.sam.predict(point_coords=input_location, point_labels=input_label, return_logits=not if_test, multimask_output=False)
        # Get target location mask
        sam_mask_out = sam_out[0, train_sam_mask]
        # Train mode
        if not if_test:
            sam_out_sigmoid = torch.nn.Sigmoid()(sam_mask_out)
            losses = self.loss(sam_out_sigmoid, loss_label.float())
            OA = evals.OA(sam_mask_out, loss_label, if_replace01=True)
        # Test mode
        else:
            sam_mask_out = sam_mask_out.int()
            # sam aug -> 1*1 conv
            sam_aug = sam_aug.detach().cpu().numpy()
            sam_aug = np.squeeze(sam_aug, axis=0)
            data_process.show_pic(sam_aug, input_location, self.per_class, None, "Argument")
            # OA
            OA = evals.OA(sam_mask_out, loss_label, if_replace01=False)
            # sam output -> final
            data_process.show_pic(sam_out, input_location, self.per_class, self.input_num, None, replace=if_test, save=if_test)
            losses = 0
        # OA
        return losses, OA

# TODO(Byan Xia): Add expression and add sth
    @torch.no_grad()
    def perdict(self):
        pass

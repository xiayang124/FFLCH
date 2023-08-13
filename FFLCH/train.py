from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from segment_anything import sam_model_registry, SamPredictor
from Simple_Net import Simple_Net
from data_process import split_location, generate_sam_label, mask_generate, test_min_data, show_pic


class model(nn.Module):
    """
    A container of model param
    """

    def __init__(self, in_channal: int, out_channal: int, param):
        super().__init__()
        self.loss = nn.MSELoss().to(param.device)
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
        :param mlp_train: Whether the mlp layer is needed,
        :param sam_train: Whether the sam layer is needed
        """
        assert mlp_train or sam_train, f"Please choose a net to use at least!"
        self.mlp_train = mlp_train
        self.sam_train = sam_train

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

    # TODO(Byan Xia): Add expression and remake with OOP
    def train_process(self, HSI: np.ndarray, Label: np.ndarray):
        # The sum of inputing SAM model label
        input_num = int(self.param.input_sam * self.param.train_num)
        # The sum of Loss label
        label_num = self.param.train_num - input_num
        for per_class in range(self.param.max_classes):
            # Get the input index and loss index
            input_location, label_location = split_location(self.train_location, input_num, label_num, per_class)
            # Get the input label and loss label
            input_label, loss_label = generate_sam_label(input_num, label_num, per_class, self.param.max_classes)
            # Get the SAM output index mask
            train_sam_mask = mask_generate(label_location, Label)

            height, width, band = HSI.shape

            loss_label = torch.from_numpy(loss_label).to(self.param.device)
            train_sam_mask = torch.from_numpy(train_sam_mask).to(self.param.device)

            assert self.param.if_mlp or self.param.if_sam, f"Wrong set of feed net!"
            if self.param.if_mlp and self.param.if_sam:
                classes = 0
                torch_train_pic = self.sam.get_apply_image(HSI.astype("uint8"))
                for epoch in range(self.param.epochs):
                    begin_time = time.time()
                    # feed net
                    losses, OA = self._mlp_sam_forward(HSI.shape, torch_train_pic, input_location, input_label, train_sam_mask, loss_label)

                    losses.backward()
                    self.Adam.step()
                    end_time = time.time()
                    print("epoch {}, loss is {}, use {} s, acc is {}".format(epoch, losses, end_time - begin_time, OA))
                    if epoch == (self.param.epochs - 1):
                        with torch.no_grad():
                            test_min = int(test_min_data(self.test_location))
                            test_input_location, _ = split_location(self.test_location, test_min, 0, per_class)
                            test_input_label, _ = generate_sam_label(test_min, 0, per_class, self.param.max_classes)
                            test_mask = mask_generate(test_input_location, Label)
                            test_mask = torch.from_numpy(test_mask).to(self.param.device)
                            test_input_label = torch.from_numpy(test_input_label).to(self.param.device)
                            test_sam_output, _, _ = self.sam.predict(input_location, input_label, return_logits=False,
                                                                     multimask_output=False)
                            show_data = test_sam_output.detach().cpu().numpy()
                            test_sam_output = test_sam_output[0, test_mask]
                            OA_losses = torch.sum(test_sam_output.int() == test_input_label)
                            OAs = OA_losses / test_sam_output.shape[0] * 100
                            show_pic(show_data, input_location, per_class, input_num)
                            print(OAs)
                classes += 1

    # TODO(Byan Xia): Add expression and add sth
    @torch.no_grad()
    def perdict(self):
        pass

    # TODO(Byan Xia): Add expression
    def _mlp_sam_forward(
            self,
            shape,
            data: torch.Tensor,
            input_location: np.ndarray,
            input_label: np.ndarray,
            train_sam_mask: np.ndarray,
            loss_label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width, band = shape

        mlp_out = self.MLP.forward(data.float())
        self.sam.set_torch_image(mlp_out, (height, width))

        sam_out, _, _ = self.sam.predict(input_location, input_label, return_logits=True, multimask_output=False)

        sam_out = sam_out[0, train_sam_mask]

        self.Adam.zero_grad()
        losses = self.loss(sam_out, loss_label.float())
        OA_mask = torch.where(sam_out > 0, 1, 0)
        same = torch.sum(OA_mask == loss_label)
        OA = same / sam_out.shape[0] * 100
        return losses, OA


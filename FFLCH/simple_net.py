import torch
import torch.nn as nn


class Simple_Net(nn.Module):
    def __init__(self, in_channels: int, out_channel=3, if_conv33=False, if_norm=True, if_hide_layer=False):
        """
        Simple Conv, used in change HSI to RGB image.
        :param in_channels: input channel
        :param out_channel: output channel
        :param if_conv33: Whether use 3*3conv, True presents use, False presents do not use, default False
        :param if_norm: Whether use norm, True presents use, False presents do not use, default True
        :param if_hide_layer: Whether use two blocks of conv, True presents use, False presents do not use, default False
        """
        super().__init__()
        self.if_conv33 = if_conv33
        self.if_norm = if_norm
        self.if_hide_layer = if_hide_layer

        if if_norm:
            self.norm = nn.BatchNorm2d(out_channel)

        # if only conv3*3
        if if_conv33 and not if_hide_layer:
            self.conv33 = nn.Conv2d(in_channels, out_channel, kernel_size=3, stride=1, padding=1)
            return

        # if hide_layer is needed:
        #     Group 1: 1*1 Conv + 3*3 Conv
        #     Group 2: 1*1 Conv + 1*1 Conv
        if if_hide_layer:
            hide_layer = 16
            self.conv11 = nn.Conv2d(in_channels, hide_layer, kernel_size=1, stride=1)
            # self.hide_norm = nn.LayerNorm(hide_layer)
            self.relu = nn.ReLU()
            # Group 1
            if if_conv33:
                self.group = nn.Conv2d(hide_layer, out_channel, kernel_size=1, stride=1)
            # Group 2
            else:
                self.group = nn.Conv2d(hide_layer, out_channel, kernel_size=1, stride=1)
            return

        # Only 1*1 Conv
        self.conv11 = nn.Conv2d(in_channels, out_channel, kernel_size=1, stride=1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # only conv3*3
        if self.if_conv33 and not self.if_hide_layer:
            data = self.conv33(data)
            if self.if_norm:
                data = self.norm(data) * 255
            return data

        # hide_layer is needed
        if self.if_hide_layer:
            data = self.conv11(data)
            # data = self.hide_norm(data)
            data = self.relu(data)
            data = self.group(data)
            if self.if_norm:
                data = self.norm(data) * 255
            return data

        # only 1*1 conv
        data = self.conv11(data)
        if self.if_norm:
            data = self.norm(data) * 255
        return data

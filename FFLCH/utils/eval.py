from typing import List

import torch
import numpy as np


def per_acc(target_output, truly, if_replace01=True, if_final=False):
    """
        The acc of per class.
        :param target_output: Target
        :param truly: Ground truth
        :param if_replace01: If replacing to 0 and 1 is needed, defalut True
        :param if_final
        :return: Per class acc
    """
    if type(target_output) == torch.Tensor:
        target_output = target_output.detach().cpu()
        truly = truly.detach().cpu()
    # data -> ndarray
    target, truly = np.array(target_output), np.array(truly)
    # if substitution is needed
    if if_replace01:
        target = np.where(target > 0, 1, 0)
    # Get the label location from truly
    foreground = np.squeeze(np.argwhere(truly == 1))
    truly_fore, target_fore = truly[foreground], target[foreground]

    same = np.sum(truly_fore == target_fore)
    acc = same / truly_fore.shape[0] * 100

    return (acc, same) if if_final else acc

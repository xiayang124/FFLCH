from typing import List

import torch
import numpy as np


class evaluation:
    def __init__(self):
        self.sames = np.array([])
        self.total = 0
        self.AA = np.array([])

    def per_acc(self, target_output, truly, if_replace01=True, if_test=False) \
            -> List[float]:
        """
        The acc of per class.
        :param if_test: Whether the mode is test
        :param target_output: Target
        :param truly: Ground truth
        :param if_replace01: If replacing to 0 and 1 is needed, defalut True
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

        foreground = np.squeeze(np.argwhere(truly == 1))
        truly_fore, target_fore = truly[foreground], target[foreground]
        if if_test:
            self.sames = np.append(self.sames, np.sum(truly_fore == target_fore))
        self.total = truly.shape[0]
        acc = np.sum(truly_fore == target_fore) / truly_fore.shape[0] * 100
        if if_test:
            self.AA = np.append(self.AA, acc)
        return acc

    def OA(self):
        return np.sum(self.sames) / self.total * 100

    def average_acc(self):
        return np.average(self.AA)

from typing import List

import torch
import numpy as np


# TODO(Byan Xia): 重做OA，完成AA
def OA(target_output, truly, if_replace01=True) \
        -> float:
    """
    Overall Accuracy(fore and background).
    :param target_output: Test output
    :param truly: True label
    :param if_replace01: Whether use substitution, defalut True(use)
    :return: OA
    """
    if type(target_output) == torch.Tensor:
        target_output = target_output.detach().cpu()
        truly = truly.detach().cpu()
    # data -> ndarray
    target = np.array(target_output)
    truly = np.array(truly)
    # if substitution is needed
    if if_replace01:
        target = np.where(target > 0, 1, 0)
    same = np.sum(target == truly)
    return same / truly.shape[0] * 100


def AA(target_output, truly, if_replace01=True)\
        -> List[float]:
    pass

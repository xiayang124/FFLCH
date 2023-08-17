import argparse
import numpy
from typing import Tuple

from data_process import read_mat


class set_args:
    """
    Get the args and data of required dataset.
    """
    def __init__(self, mat_name: str):
        self._mat_name = mat_name.lower()

    def _parse(self):
        """
        Set the parse of dataset.
        :return: dataset args
        """
        params = argparse.ArgumentParser(description="Training Process Params")
        params.add_argument("--hsi", type=str, default="./dataset/hsi/Pavia.mat", help="HSI dir")
        params.add_argument("--label", type=str, default="./dataset/label/Pavia_gt.mat", help="Lable dir")
        params.add_argument("--device", type=str, default="cuda:0", help="choose device for training")
        params.add_argument("--checkpoint_path", type=str, default="../segment_anything/sam_vit_b_01ec64.pth",
                            help="sam model path, must be same as model")
        params.add_argument("--mode", type=str, default="vit_b", help="sam model, must be same as checkpoint_path")
        params.add_argument("--batch_size", type=int, default=1, help="batch_size")
        params.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
        params.add_argument("--train_num", type=int, default=4, help="the num of one class train data")
        params.add_argument("--max_classes", type=int, default=0)
        params.add_argument("--input_sam", type=float, default=0.5, help="the rate of train point to sam")
        params.add_argument("--epochs", type=int, default=100, help="epoch")
        params.add_argument("--if_sam", type=bool, default=True, help="if use sam")
        params.add_argument("--if_mlp", type=bool, default=True, help="if use mlp")
        if self._mat_name == "pavia":
            self._args = params.parse_args()
            self._args.checkpoint_path = "../segment_anything/sam_vit_h_4b8939.pth"
            self._args.mode = "vit_h"

    def _load_data(self)\
            -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Get required mat data, label data and set max_classes.
        """
        self._label = read_mat(self._args.label)
        self._mat = read_mat(self._args.hsi)
        self._args.max_classes = numpy.max(self._label)

    def get_arg(self):
        """
        Get required dataset's args, mat data and label data.
        :return: args, HSI data, label data
        """
        self._parse()
        self._load_data()
        return self._args, self._mat, self._label

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as io
import argparse


def read_mat(mat_dir: str) -> np.ndarray:
    """
    Get the ndarray about mat.
    :param mat_dir: dir about mat file
    :return: numpy_data: mat data with ndarray
    """
    mat = io.loadmat(mat_dir)
    for data in mat.values():
        if type(data) == np.ndarray:
            numpy_data = np.array(data)
            return numpy_data
    raise Exception("Wrong mat_dir or Wrong mat file!")


class data_aug:
    def __init__(self, HSI: np.ndarray, Label: np.ndarray, max_class: int, train_num: int):
        self.HSI = HSI
        self.Label = Label
        self.max_class = max_class
        self.train_num = train_num
        # init container
        self.train_location = {classes: "" for classes in range(self.max_class)}
        self.test_location = {classes: "" for classes in range(self.max_class)}
        self.test_total_num = 0
        self.test_data_num = np.empty(shape=(0, ))

    def segment_data(self):
        """
        Get non-zero data and segment data from initial HSI and Labels.
        :return: Train location, Test location
        """
        # Copy data
        hsi = self.HSI.copy()
        label = self.Label.copy()
        # basic assert
        assert np.array(label.shape).shape != 2, f"Wrong dim input about label!"
        assert np.array(hsi.shape).shape != 3, f"Wrong dim input about HSI!"

        # make copy version
        mat = hsi.copy()
        mat_label = label.copy()

        height, width, band = mat.shape
        mat = mat.reshape(-1, band)

        # init container
        for per_class in range(self.max_class):
            # same label position
            per_class_location = np.argwhere(mat_label == per_class + 1)

            label_num, _ = per_class_location.shape
            test_num = label_num - self.train_num
            # shuffle location
            np.random.shuffle(per_class_location)
            # train location and test location install in container
            self.train_location[per_class] = per_class_location[0:self.train_num, :]
            self.test_location[per_class] = per_class_location[self.train_num: -1, :]
        return self.train_location, self.test_location

    def set_test_data_num(self):
        """
        Set the num of all class test location and the min of all class num(No Return).
        """
        for per_class in self.test_location.values():
            current_class_num = per_class.shape[0]
            self.test_total_num += current_class_num
            self.test_data_num = np.append(self.test_data_num, current_class_num)
        return None


def set_band(HSI: np.ndarray, generate_num: int, mode="random") -> np.ndarray:
    """
    The way of exchange mutil band to 3 band, defalut random.
    :param HSI: Initial HSI
    :param generate_num: Num of generate
    :param mode: The way to change, defalut random
    :return: segment
    """
    choose_band = 3
    # Deep Copy
    hsi = HSI.copy()
    height, width, band = HSI.shape
    # Initial Data Group
    segment = np.zeros((generate_num, 3, height, width))
    if mode == "random":
        # Exchange dim to shuffle dim
        hsi = hsi.transpose((2, 0, 1))
        for num in range(generate_num):
            np.random.shuffle(hsi)
            # Choose the 0 to choose band
            hsi = hsi[0: choose_band, :, :]
            # Put into container
            segment[num, :, :, :] = hsi
        # Recover
        segment = segment.transpose((0, 2, 3, 1))
    return segment

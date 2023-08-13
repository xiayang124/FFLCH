from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as io
import argparse
from sklearn.decomposition import PCA


def read_mat(mat_dir: str) \
        -> np.ndarray:
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


def set_pca(HSI: np.ndarray, choose_band=3) \
        -> np.ndarray:
    """
    Use PCA to get target hsi data.
    :param choose_band: The principal components which remain
    :param HSI: Initial HSI
    :return: PCA output
    """
    # Deep Copy
    hsi = HSI.copy()
    heigth, width, band = hsi.shape
    # Reshape to pair pca input
    hsi = hsi.reshape((heigth * width), band)
    # PCA process
    pca = PCA(n_components=choose_band, copy=False)
    hsi = pca.fit_transform(hsi)
    # Reshape as initial image
    hsi = hsi.reshape(heigth, width, choose_band)
    return hsi


def split_location(label_location: dict, input_num: int, label_num: int, current_class: int) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the splited location as the given num. When input test data min label num, this function can
    generate test location.

    :param label_num: The num of label location
    :param label_location: Dictionary about train dataset
    :param input_num: The num of input SAM location
    :param current_class: current train class
    :return: input_index, label_index
    """

    # Initial input_index and label_index
    input_index = np.empty(shape=(0, 2))
    label_index = np.empty(shape=(0, 2))
    for per_class in label_location.values():
        label_num, _ = per_class.shape
        # Shuffle location and input_index
        np.random.shuffle(per_class)
        input_index = np.append(input_index, per_class[0: input_num, :], axis=0)
        label_index = np.append(label_index, per_class[input_num:, :], axis=0)
    return input_index, label_index


def generate_sam_label(input_num: int, loss_num: int, current_class: int, max_class: int) \
        -> np.ndarray:
    """
    Acorrding to the class which is training at present to generate the label that can
    distinguish the background and fore.
    :param loss_num: The num of loss pair label
    :param input_num: The num of input SAM label
    :param current_class: current train class
    :param max_class: The num of class
    :return: input_label, loss_label
    """
    count = 0
    # Initial label containers
    input_label = np.empty(shape=(0, ))
    loss_label = np.empty(shape=(0, ))

    for per_class in range(max_class):
        # if Ture, append 1 (fore)
        if current_class == count:
            input_label = np.append(input_label, np.ones(shape=(input_num, )), axis=0)
            loss_label = np.append(loss_label, np.ones(shape=(loss_num, )), axis=0)
            count += 1
            continue
        # Others append 0 (background)
        input_label = np.append(input_label, np.zeros(shape=(input_num, )), axis=0)
        loss_label = np.append(loss_label, np.zeros(shape=(loss_num, )), axis=0)
    return input_label, loss_label


def mask_generate(loss_location: np.ndarray, Lable: np.ndarray)\
        -> np.ndarray:
    """
    Generate the mask of SAM output.
    :param loss_location: Location of loss pair
    :param Lable: Initial label
    :return: SAM output mask(bool)
    """
    # Deep Copy
    labels = Lable.copy()
    max_class = np.max(labels)
    label_num, _ = loss_location.shape
    for per_label in range(label_num):
        # Get per label location
        current_label_x, current_label_y = loss_location[per_label, :].astype("int")
        # Set location in label to -1
        labels[current_label_x, current_label_y] = (max_class + 2)
    # Set label where equals -1 to True, Others False
    label = np.where(labels == (max_class + 2), True, False)
    return label


# TODO(Byan Xia): Add expression
def test_min_data(test_location: dict):
    class_num = np.empty(shape=(0, ))
    for per_class in test_location.values():
        per_class_num, _ = per_class.shape
        class_num = np.append(class_num, per_class_num)
    return np.min(class_num)


# TODO(Byan Xia): Add expression
def show_pic(pic, location, current_label, get_num):
    label_num, _ = location.shape
    pic = pic.transpose((1, 2, 0))
    pic = np.where(pic == True, 255, 0)

    plt.subplot(111)
    plt.imshow(pic)
    for per_label in range(label_num):
        x, y = location[per_label]
        if int(per_label / get_num) == current_label:
            plt.plot(y, x, c='r', marker='x')
        else:
            plt.plot(y, x, c='g', marker='x')
    plt.show()

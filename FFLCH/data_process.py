import argparse
import gc
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import torch
from sklearn.decomposition import PCA
import random


def read_mat(mat_dir: str) \
        -> np.ndarray:
    """
    Get the ndarray about mat.
    -----------------------------------------
    Arguments:
        mat_dir (str): dir about mat file
    -----------------------------------------
    Returns:
        (np.ndarray): mat data with ndarray
    """
    mat = io.loadmat(mat_dir)
    for data in mat.values():
        if type(data) == np.ndarray:
            numpy_data = np.array(data)
            return numpy_data
    raise Exception("Wrong mat_dir or Wrong mat file!")


def segment_data(HSI, Label, train_num):
    """
    Get non-zero data and segment data from initial HSI and Labels.
    -----------------------------------------------------------------
    Arguments:
        HSI: Initial data
        Label: Initial label
        train_num: The num for training
    -----------------------------------------------------------------
    Returns:
        (dict): The location of train data.
        (dict): The location of test data.
    """
    # make copy version
    mat = HSI.copy()
    mat_label = Label.copy()

    assert np.array(mat_label.shape).shape != 2, f"Wrong dim input about label!"
    assert np.array(mat.shape).shape != 3, f"Wrong dim input about HSI!"

    max_classes = np.max(mat_label)

    train_location = {per: 0 for per in range(max_classes)}
    test_location = {per: 0 for per in range(max_classes)}

    height, width, band = mat.shape
    mat = mat.reshape(-1, band)
    # init container
    for per_class in range(max_classes):
        # same label position
        per_class_location = np.argwhere(mat_label == per_class + 1)
        label_num, _ = per_class_location.shape
        test_num = label_num - train_num
        # shuffle location
        np.random.shuffle(per_class_location)
        # train location and test location install in container
        train_location[per_class] = per_class_location[0:train_num, :][:, [1, 0]]
        test_location[per_class] = per_class_location[train_num: -1, :][:, [1, 0]]
    return train_location, test_location


def set_pca(HSI: np.ndarray, choose_band=3) \
        -> np.ndarray:
    """
    Use PCA to get target hsi data.
    --------------------------------------------------------
    Arguments:
        choose_band: The principal components which remain.
        HSI: Initial HSI.
    --------------------------------------------------------
    Returns:
        (np.ndarray): PCA output.
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


def show_pic(data, location: np.ndarray, current_label: int, get_num: int, stage: str, replace=False, save=False):
    """
    Show the final data.
    ----------------------------------------------------------
    :param data: Pic
    :param location: The point location
    :param current_label: The label type which is training
    :param get_num: Per class train num
    :param stage: Where the pic comes from
    :param replace: Turn the Bool data to Binary data
    :param save: Whether save the file
    """
    # exchange
    if type(data) == torch.Tensor:
        data = data.detach().cpu()
    target_pic = np.array(data)
    # Norm to [0, 255]
    if not replace:
        target_pic = data_to_255(target_pic)
    target_pic = np.squeeze(target_pic)
    # The shape of pic
    dim = target_pic.shape
    # The num of the location point
    label_num, _ = location.shape
    # Draw the location point
    for per_label in range(label_num):
        x, y = location[per_label]
        if int(per_label / get_num) == current_label:
            plt.plot(x, y, c='r', marker='x')
        else:
            plt.plot(x, y, c='g', marker='x')

    assert len(dim) == 2 or len(dim) == 3, f"Require 2/3 dim, expected {len(dim)}"
    if len(dim) == 2:
        # The segment data
        plt.subplot(111)
        # Exchange the data to float
        target_pic = target_pic.astype("float")
        # Show the pic
        plt.imshow(target_pic)

        if replace:
            target_pic = np.where(target_pic == True, 255, 0)
        # DO NOT SET BLOCK TO FALSE!
        plt.show(block=True)
        save = random.randint(1, 10000000)
        plt.imsave("./testimage/test" + str(save) + ".png", target_pic)
        plt.close()
        gc.collect()
        return
    else:
        dim_1, dim_2, dim_3 = target_pic.shape
        save = random.randint(1, 10000000)
        # Reshape back to normal pic
        if dim_1 == 3:
            target_pic = target_pic.transpose((1, 2, 0))
        if replace:
            target_pic = np.where(target_pic == True, 255, 0)
        # Float
        target_pic = target_pic.astype("float")
        target_pic = target_pic / 255
        plt.imshow(target_pic)
        plt.show(block=True)
        # Save file
        if save:
            plt.imsave("./" + stage + str(save) + ".png", target_pic)
            plt.close()
        gc.collect()
        return


def data_to_255(HSI) \
        -> np.ndarray:
    """
    Norm data and set range to [0, 255]
    :param HSI: Data
    :return: Norm to [0, 255] data
    """
    return (HSI - np.min(HSI)) / (np.max(HSI) - np.min(HSI)) * 255


class data_argument:
    def __init__(
            self,
            pca_HSI: np.ndarray,
            label: np.ndarray,
            train_location: dict,
            test_location: dict,
            params: argparse.Namespace
    ):
        self.HSI = pca_HSI
        self.Label = label
        self.params = params
        self.train_num = params.train_num
        self.input_sam = params.input_sam
        self.train_location = train_location
        self.test_location = test_location
        self.max_classes = np.max(label)

    def get_input_num(
            self
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The num of input sam data and Loss data.
        :returns: input_num, label_num
        """
        # The sum of inputing SAM model label
        self.input_num = int(self.train_num * self.input_sam)
        # The sum of Loss label
        self.label_num = self.train_num - self.input_num
        return self.input_num, self.label_num

    def split_location(
            self,
            input_num=0,
            if_test=False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the splited location according the given input sam model num. When input test data min label num,
        this function can generate test location.

        Arguments:
            input_num: The data of input index num, if if_test is Ture, this argument MUST fill
            if_test: The mode of this function, if True generate test data, defalut False

        Returns:
            input_index
            label_index
        """
        # Initial input_index and label_index
        input_index = np.empty(shape=(0, 2))
        label_index = np.empty(shape=(0, 2))
        # Train
        if not if_test:
            label_location = self.train_location
            cut = self.input_num
        # Test
        else:
            label_location = self.test_location
            assert input_num != 0 and type(input_num) == int, f"Argument input_num must have a valid int"
            cut = input_num
        # For each class
        for per_class in label_location.values():
            label_num, _ = per_class.shape
            # Shuffle location and input_index
            np.random.shuffle(per_class)
            input_index = np.append(input_index, per_class[0: cut, :], axis=0)
            label_index = np.append(label_index, per_class[cut:, :], axis=0)
        return input_index, label_index

    def generate_sam_label(
            self,
            test_min_num=0,
            if_test=False
    ) -> np.ndarray:
        """
        Acorrding to the class which is training at present to generate the label that can
        distinguish the background and foreground.
        ------------------------------------------------------------------------------------------------
        Arguments:
            test_min_num: The data of test per class num, if if_test is Ture, this argument MUST fill
            if_test: The mode of this function, if True generate test data, defalut False
        ------------------------------------------------------------------------------------------------
        :return: input_label_can, loss_label_can
        """
        # The num of per class label length
        if not if_test:
            input_label_length = self.input_num * self.max_classes
            loss_label_length = self.label_num * self.max_classes
            input_num = self.input_num
        else:
            assert test_min_num != 0 and type(test_min_num) == int, f"Argument test_min_num must have a valid int"
            input_label_length = test_min_num * self.max_classes
            input_num = test_min_num
            loss_label_length = 0
        # Initial zero data
        input_label = np.zeros(shape=(input_label_length,))
        loss_label = np.zeros(shape=(loss_label_length,))
        # Initial dict
        input_label_can = {i: input_label.copy() for i in range(self.max_classes)}
        loss_label_can = {i: loss_label.copy() for i in range(self.max_classes)}
        # Set foreground
        for per_class in range(self.max_classes):
            input_label_can[per_class][per_class * input_num: (per_class + 1) * input_num, ] = 1
            if not if_test:
                loss_label_can[per_class][per_class * self.label_num: (per_class + 1) * self.label_num, ] = 1
        return input_label_can, loss_label_can

    def mask_generate(
            self,
            loss_location: np.ndarray
    ) -> np.ndarray:
        """
        Generate the mask of SAM output.
        --------------------------------------------------------
        :param loss_location: Location of loss pair
        :return: SAM output mask(bool)
        """
        # Deep Copy
        labels = self.Label.copy()
        label_num, _ = loss_location.shape
        for per_label in range(label_num):
            # Get per label location
            current_label_x, current_label_y = loss_location[per_label, :].astype("int")
            # Set location in label to -1
            labels[current_label_y, current_label_x] = (self.max_classes + 1)
        # Set label where equals -1 to True, Others False
        label = np.where(labels == (self.max_classes + 1), True, False)
        return label

    def test_num(self):
        """
        Get test data per class num and min num.
        """
        # Initial data
        class_num = np.empty(shape=(0,))

        for per_class in self.test_location.values():
            per_class_num, _ = per_class.shape
            class_num = np.append(class_num, per_class_num)
        return int(np.min(class_num)), class_num

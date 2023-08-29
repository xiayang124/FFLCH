import gc
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import torch
from sklearn.decomposition import PCA


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


def show_pic(data: torch.Tensor or np.ndarray, location: np.ndarray, current_label: int, get_num: int, epoch: int,
             stage: str, replace=False):
    """
    Show the final data.
    ----------------------------------------------------------
    :param epoch: Current epoch
    :param data: Pic
    :param location: The point location
    :param current_label: The label type which is training
    :param get_num: Per class train num
    :param stage: Where the pic comes from
    :param replace: Turn the Bool data to Binary data
    """
    # exchange
    if type(data) == torch.Tensor:
        data = data.detach().cpu()
    target_pic = np.array(data).astype("float")
    # Norm to [0, 255]
    if not replace:
        target_pic = data_to_255(target_pic) / 255
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

    assert len(dim) == 2 or len(dim) == 3, f"Require 2/3 dims, but get {len(dim)} dims."
    # The segment data
    plt.subplot(111)
    if dim[0] == 3:
        target_pic = target_pic.transpose((1, 2, 0))
    if replace:
        target_pic = np.where(target_pic == True, 255, 0)
    # Show the pic
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(target_pic)
    plt.savefig("./image out/" + stage + "/class" + str(current_label) + " " + str(epoch) + ".png")
    plt.show(block=True)
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


def write_file(current_label: int, OA: float):
    if current_label == 0:
        with open(r"./pth/record.txt", 'a') as file:
            lines = "\nFinal Result"
            file.writelines(lines)
    with open(r"./pth/record.txt", 'a') as file:
        lines = "\nClass " + str(current_label + 1) + " OA is " + str(OA)
        file.writelines(lines)


def location_seg(Label: np.ndarray, train_num: int, input_rate: float) \
        -> Tuple[dict, dict, dict, dict]:
    """
    Get location about input sam, loss, all train data and test data.
    :param Label: Initial label data.
    :param train_num: The sum of train data
    :param input_rate: The sum of input segment anything model
    :return: input_location, loss_location, train_location, test_location
    """
    # Deep copy
    label = Label.copy()
    # Class num
    max_classes = np.max(Label)
    # The sum of input sam
    input_num = int(train_num * input_rate)
    # Initial dict
    input_location = {classes: 0 for classes in range(max_classes)}
    loss_location = {classes: 0 for classes in range(max_classes)}
    train_location = {classes: 0 for classes in range(max_classes)}
    test_location = {classes: 0 for classes in range(max_classes)}

    for per_class in range(max_classes):
        # Get location
        per_class_location = np.argwhere(label == per_class + 1)
        np.random.shuffle(per_class_location)
        # Index ([:, [1, 0] means exchange axis x and axis y, row and line cannot use in uns)
        input_location[per_class] = per_class_location[0: input_num, :][:, [1, 0]]
        loss_location[per_class] = per_class_location[input_num: train_num, :][:, [1, 0]]
        train_location[per_class] = per_class_location[0: train_num, :][:, [1, 0]]
        test_location[per_class] = per_class_location[train_num:, :][:, [1, 0]]
    return input_location, loss_location, train_location, test_location


def get_label(location: dict) \
        -> Dict:
    """
    Get the location label.
    :param location: Per class location
    :return: data_label
    """
    # Class sum
    max_classes = len(location.keys())
    # Initial per class
    per_class_len = np.empty(shape=(0, ), dtype=int)
    # Per class num
    for per_class in location.values():
        per_class_len = np.append(per_class_len, np.array(per_class.shape[0]))
    # Initial zeros label
    zero_label = np.zeros(shape=(np.sum(per_class_len), ))
    data_label = {classes: zero_label.copy() for classes in range(max_classes)}

    index_begin = 0
    index_end = 0
    # Set corresponding class to foreground label (set to 1)
    for cls in range(max_classes):
        index_end += per_class_len[cls]
        data_label[cls][index_begin: index_end] = 1
        index_begin += per_class_len[cls]
    return data_label


def concat_dict(dic: dict)\
        -> np.ndarray:
    """
    Concat per class.
    :param dic: Target dict
    :return: concated
    """
    concated = np.empty(shape=(0, 2))
    for per_class in dic.values():
        concated = np.concatenate((concated, per_class))
    return concated

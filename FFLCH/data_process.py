import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as io


def read_mat(mat_dir: str):
    mat = io.loadmat(mat_dir)
    for data in mat.values():
        if type(data) == np.ndarray:
            numpy_data = np.array(data)
            return numpy_data
    raise Exception("Wrong mat_dir or Wrong mat file!")

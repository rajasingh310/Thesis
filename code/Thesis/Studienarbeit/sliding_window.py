from bilateral_weight_matrix import bilateral_weight_matrix
import numpy as np

import math


def sliding_window(image, window_row, window_colm, window_stride, std_dev):
    sliding_window_images = np.empty(
        (math.floor((image.shape[0] - window_row + 1) / window_stride), math.floor((image.shape[1] - window_colm + 1) / window_stride)),
        dtype=np.ndarray)
    bilateral_weight_images = np.empty(
        (math.floor((image.shape[0] - window_row + 1) / window_stride), math.floor((image.shape[1] - window_colm + 1) / window_stride)),
        dtype=np.ndarray)

    for row1 in range(0, image.shape[0] - window_row + 1, window_stride):
        row2 = row1 + window_row
        if row2 > image.shape[0]:
            break

        for col1 in range(0, image.shape[1] - window_colm + 1, window_stride):
            col2 = col1 + window_colm
            if col2 > image.shape[1]:
                break
            sliding_window_images[row1, col1] = image[row1:row2, col1:col2, :, :]
            bilateral_weight_images[row1, col1] = bilateral_weight_matrix(sliding_window_images[row1, col1], 7.5, std_dev)

    return sliding_window_images, bilateral_weight_images
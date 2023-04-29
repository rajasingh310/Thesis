import numpy as np

from sliding_window import sliding_window
from knn import knn


def images_and_labels(new_dataset, window_row, window_colm, window_stride, mat_names, k, std_dev):
    img = np.empty((new_dataset.shape[0]), dtype=np.ndarray)
    raw_img = np.empty((new_dataset.shape[0]), dtype=np.ndarray)

    raw_images_data = np.empty(
        (new_dataset.shape[0], new_dataset.shape[1]),
        dtype=np.ndarray)
    bilateral_images_data = np.empty(
        (new_dataset.shape[0], new_dataset.shape[1]),
        dtype=np.ndarray)

    for mat_idx in range(len(new_dataset)):
        for inst_idx in range(len(new_dataset[mat_idx])):
            image = new_dataset[mat_idx][inst_idx]
            [raw_images_data[mat_idx, inst_idx], bilateral_images_data[mat_idx, inst_idx]] = sliding_window(image, window_row, window_colm,
                                                                                                            window_stride, std_dev)
            bilateral_images_data[mat_idx, inst_idx] = bilateral_images_data[mat_idx, inst_idx].flatten()  # convert a matrix into vector
            raw_images_data[mat_idx, inst_idx] = raw_images_data[mat_idx, inst_idx].flatten()  # convert a matrix into vector

        img[mat_idx] = np.concatenate(bilateral_images_data[mat_idx, :], axis=0)
        img[mat_idx] = np.stack(img[mat_idx][:], axis=3)
        img[mat_idx] = np.expand_dims(img[mat_idx], axis=2)
        img[mat_idx] = np.transpose(img[mat_idx], (4, 0, 1, 2, 3))
        print(mat_idx)

        raw_img[mat_idx] = np.concatenate(raw_images_data[mat_idx, :], axis=0)
        raw_img[mat_idx] = np.stack(raw_img[mat_idx][:], axis=4)
        raw_img[mat_idx] = np.transpose(raw_img[mat_idx], (4, 0, 1, 2, 3))
        print(mat_idx)

    labels_data = np.empty((0, 0))  # initialize as empty numpy array
    for mat_idx in range(len(mat_names)):
        labels_data_mat = np.tile(mat_names[mat_idx], (img[mat_idx].shape[0], 1))
        if labels_data.size == 0:  # if this is the first iteration, assign the first array to labels_data
            labels_data = labels_data_mat
        else:  # concatenate the new array with the existing array
            labels_data = np.concatenate((labels_data, labels_data_mat))

    images = np.vstack(img[:])
    images = np.transpose(images, (1, 2, 3, 4, 0))

    raw_images = np.vstack(raw_img[:])
    raw_images = np.transpose(raw_images, (1, 2, 3, 4, 0))

    images = knn(images, raw_images, k)
    labels = np.vstack(labels_data)

    return images, labels

import numpy as np

def find_sigma1(data):
    sigma = np.empty((data.shape[0], ), dtype=object)
    for i in range(data.shape[0]):
        sigma_i = []
        for j in range(data.shape[1]):
            data_ij = np.reshape(data[i, j], (8, 2, -1), order='F')
            data_ij = np.transpose(data_ij, (2, 0, 1))
            data_ij = np.reshape(data_ij, (data_ij.shape[0], -1), order='F')
            sigma_i.append(data_ij)
        sigma[i] = np.vstack(sigma_i) - np.min(np.vstack(sigma_i))
    sigma = std_deviation(sigma)

    mat = np.empty((sigma.shape[0], ), dtype=object)
    for i in range(sigma.shape[0]):
        mat[i] = sigma[i]
        mat[i] = mat[i].reshape((1, 16))

    matrix = np.vstack(mat[:])

    # Calculate the mean of each column
    mean_of_columns = np.mean(matrix, axis=0)
    mean_of_columns = mean_of_columns.reshape((1, 1, 8, 2))

    return mean_of_columns


def std_deviation(sigma):
    std_dev = np.empty((sigma.shape[0], ), dtype=object)
    for i in range(sigma.shape[0]):
        std_dev[i] = np.std(sigma[i], axis=0)
    return std_dev
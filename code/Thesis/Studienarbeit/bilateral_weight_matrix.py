import numpy as np


def bilateral_weight_matrix(image, sigmaD, sigmaR):

    # Compute the center pixel of the image
    centreR = np.ceil(image.shape[0]/2).astype(int)
    centreC = np.ceil(image.shape[1]/2).astype(int)

    # Compute the spatial weight matrix
    r = np.tile(np.arange(1, image.shape[0]+1).reshape(-1, 1), (1, image.shape[1]))
    c = np.tile(np.arange(1, image.shape[1]+1), (image.shape[0], 1))

    r = r - centreR
    c = c - centreC

    r = r**2
    c = c**2

    # Compute the distance matrix
    dist = np.sqrt(r+c)
    # Compute the range matrix
    b_image = image - np.tile(image[centreR-1, centreC-1, :, :], (image.shape[0], image.shape[1], 1, 1))
    b_image = (b_image**2)/(2*sigmaR**2)
    b_matrix = np.sum(b_image, axis=2)

    # Compute the bilateral weight matrix
    b_matrix = np.exp(-np.tile((dist/(2*sigmaD**2))[:, :, np.newaxis], (1, 1, 2)) - b_matrix)


    return b_matrix
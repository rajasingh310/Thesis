import numpy as np


def knn(images, raw_images, k):
    centreR = np.ceil(images.shape[0] / 2).astype(int)
    centreC = np.ceil(images.shape[1] / 2).astype(int)

    r = np.repeat(np.arange(images.shape[0])[:, np.newaxis], images.shape[1], axis=1).ravel()
    c = np.tile(np.arange(images.shape[1]), images.shape[0]).ravel()

    a = np.empty((images.shape[4], 1), dtype=object)
    for n in range(images.shape[4]):

        b_image = images[:, :, :, :, n] - np.tile(images[centreR - 1, centreC - 1, :, :, n], (images.shape[0], images.shape[1], 1, 1))
        b_image = np.squeeze(b_image)
        b_image = np.linalg.norm(b_image, ord=2, axis=2)
        b_image = np.column_stack((b_image.ravel(), r, c))

        b_image = b_image[b_image[:, 0].argsort()]
        i = b_image[:k, 1]
        j = b_image[:k, 2]

        x = np.empty((1, k), dtype=object)  # extracting top k nearest 3D-pixels
        for l in range(k):
            x[0, l] = raw_images[int(i[l]), int(j[l]), :, :, n]
            x[0, l] = np.reshape(x[0, l], (1, 1, 8, 2))
        a[n, 0] = np.concatenate(x[0, :], axis=1)

    knn_images = np.stack(a[:, 0], axis=4)

    return knn_images
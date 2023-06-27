import numpy as np


class Padding:
    def __init__(self, dataset, pad=True):

        self.new_dataset = np.empty((dataset.shape[0], dataset.shape[1]), dtype=object)
        i_res = (8, 172, 224)

        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[1]):

                # data orientation correction (8, 172, 224) from (172, 224, 8)
                dataset[i, j] = np.rot90(dataset[i, j], 1, axes=(2, 0))
                dataset[i, j] = np.rot90(dataset[i, j], 1, axes=(2, 1))

                if pad:

                    shape = dataset[i, j].shape
                    r_diff = i_res[1] - shape[1]
                    c_diff = i_res[2] - shape[2]

                    r_diff_1 = r_diff // 2
                    r_diff_2 = r_diff - r_diff_1

                    c_diff_1 = c_diff // 2
                    c_diff_2 = c_diff - c_diff_1

                    self.new_dataset[i, j] = np.pad(dataset[i, j], ((0, 0), (r_diff_1, r_diff_2), (c_diff_1, c_diff_2)), mode='symmetric')

                else:

                    self.new_dataset[i, j] = dataset[i, j]

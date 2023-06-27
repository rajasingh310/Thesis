import numpy as np


class HandleDataImbalance:
    def __init__(self, dataset, dataBalance=True):
        new_dataset = np.empty(dataset.shape[0], dtype=object)
        mat_sample_size = []
        for mat in range(dataset.shape[0]):
            for inst in range(dataset.shape[1]):
                dataset[mat, inst] = np.rot90(dataset[mat, inst], 1, axes=(2, 0))
                dataset[mat, inst] = np.rot90(dataset[mat, inst], 1, axes=(2, 1))
                dataset[mat, inst] = np.reshape(dataset[mat, inst], (dataset[mat, inst].shape[0], 1, -1))
            new_dataset[mat] = np.concatenate((dataset[mat, :]), axis=2)
            mat_sample_size.append(new_dataset[mat].shape[2])

            #  shuffle dataset
            permutations = np.random.permutation(new_dataset[mat].shape[2])
            new_dataset[mat] = new_dataset[mat][:, :, permutations]

        if dataBalance:
            self.balanced_dataset = np.empty(dataset.shape[0], dtype=object)
            for mat in range(dataset.shape[0]):
                for inst in range(dataset.shape[1]):
                    self.balanced_dataset[mat] = new_dataset[mat][:, :, :min(mat_sample_size)]

        else:
            self.balanced_dataset = new_dataset


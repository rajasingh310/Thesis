import numpy as np

class FeatureExtraction:

    def __init__(self, dataset):

        n_mats = dataset.shape[0]
        n_inst = dataset.shape[1]


        self.abs_new_dataset = np.empty((n_mats, n_inst), dtype=object)
        self.r_i_new_dataset = np.empty((n_mats, n_inst), dtype=object)

        for i in range(n_mats):
            for j in range(n_inst):

                abs_value = np.abs(dataset[i, j])
                # Calculate the mean along the second axis (axis=1)
                mean_abs_value = np.mean(abs_value, axis=0)

                # Extract the real and imaginary components of the data
                real_value = np.real(dataset[i, j])[np.newaxis, ...]
                imag_value = np.imag(dataset[i, j])[np.newaxis, ...]


                # Stack the real and imaginary components along the channel axis
                self.abs_new_dataset[i, j] = mean_abs_value
                self.r_i_new_dataset[i, j] = np.concatenate([real_value, imag_value], axis=0)


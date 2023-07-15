import numpy as np

class Normalization:
    def __init__(self, data, pix_ranges, poi):    # poi is pixel of interest: Boolean
        self.new_dataset = self.data_normalization(data, pix_ranges, poi)

    def data_normalization(self, data, pix_ranges, poi):

        dataset = data['dataset']
        amp_imgs = data['amp_imgs']
        depth_imgs = data['depth_imgs']
        f_0 = data['f_0']
        c = data['c']

        n_mats, n_inst_per_mat = dataset.shape
        new_dataset = np.empty((n_mats, n_inst_per_mat), dtype=object)

        # Add a row of zeros at the beginning
        pix_ranges = np.vstack([np.zeros((1, pix_ranges.shape[1])), pix_ranges])
        # Add a column of zeros at the beginning
        pix_ranges = np.hstack([np.zeros((pix_ranges.shape[0], 1)), pix_ranges])

        for mat_idx in range(1, n_mats + 1):
            for inst_idx in range(1, n_inst_per_mat + 1):

                # Obtain reflectivity- and depth-independent features from raw data
                amp_data = amp_imgs[mat_idx - 1, inst_idx - 1]
                amp_norm_dataset = dataset[mat_idx - 1, inst_idx - 1][:, :, 1:] / amp_data[:, :, np.newaxis]

                # Back calculating phase from depth information
                phase_data = 2 * (2 * np.pi) * f_0 * 1e6 * depth_imgs[mat_idx - 1, inst_idx - 1] / c
                amp_size = amp_norm_dataset.shape
                phase = np.reshape(np.arange(1, amp_size[2] + 1) * 1j, (1, 1, amp_size[2]))
                phase = np.tile(phase, (amp_size[0], amp_size[1], 1))
                phase *= np.tile(np.reshape(phase_data, (amp_size[0], amp_size[1], 1)), (1, 1, amp_size[2]))
                new_dataset[mat_idx - 1, inst_idx - 1] = amp_norm_dataset / np.exp(phase)
                # Cropping Region of Interest from Images

                if (poi):

                    # Retrieve pixel range (only data within "valid" pixel ranges is considered)
                    init_row_idx = int(pix_ranges[2 * (inst_idx - 1) + 1, (2 * (mat_idx - 1) + 1)])
                    final_row_idx = int(pix_ranges[2 * (inst_idx - 1) + 1, (2 * mat_idx)])
                    init_col_idx = int(pix_ranges[(2 * inst_idx), (2 * (mat_idx - 1) + 1)])
                    final_col_idx = int(pix_ranges[(2 * inst_idx), (2 * mat_idx)])

                    new_dataset[mat_idx - 1, inst_idx - 1] = new_dataset[mat_idx - 1, inst_idx - 1][init_row_idx - 1:final_row_idx,
                                                             init_col_idx - 1:final_col_idx, :]

        return new_dataset

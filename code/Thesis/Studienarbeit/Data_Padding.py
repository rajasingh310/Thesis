import numpy as np


class Padding:
    def __init__(self, dataset, pix_ranges, pad=True):

        n_mats = dataset.shape[0]
        n_inst_per_mat = dataset.shape[1]
        i_res = (8, 172, 224)

        self.new_dataset = np.empty(dataset.shape, dtype=object)

        # Add a row of zeros at the beginning
        pix_ranges = np.vstack([np.zeros((1, pix_ranges.shape[1])), pix_ranges])
        # Add a column of zeros at the beginning
        pix_ranges = np.hstack([np.zeros((pix_ranges.shape[0], 1)), pix_ranges])

        for mat_idx in range(1, n_mats + 1):
            for inst_idx in range(1, n_inst_per_mat + 1):

                # data orientation correction (8, 172, 224) from (172, 224, 8)
                dataset[mat_idx - 1, inst_idx - 1] = np.rot90(dataset[mat_idx - 1, inst_idx - 1], 1, axes=(2, 0))
                dataset[mat_idx - 1, inst_idx - 1] = np.rot90(dataset[mat_idx - 1, inst_idx - 1], 1, axes=(2, 1))

                if pad:

                    # Retrieve pixel range (only data within "valid" pixel ranges is considered)
                    init_row_idx = int(pix_ranges[2 * (inst_idx - 1) + 1, (2 * (mat_idx - 1) + 1)])
                    final_row_idx = int(pix_ranges[2 * (inst_idx - 1) + 1, (2 * mat_idx)])
                    init_col_idx = int(pix_ranges[(2 * inst_idx), (2 * (mat_idx - 1) + 1)])
                    final_col_idx = int(pix_ranges[(2 * inst_idx), (2 * mat_idx)])

                    r_diff_1 = init_row_idx - 1
                    r_diff_2 = i_res[1] - final_row_idx
                    c_diff_1 = init_col_idx - 1
                    c_diff_2 = i_res[2] - final_col_idx

                    self.new_dataset[mat_idx - 1, inst_idx - 1] = np.pad(dataset[mat_idx - 1, inst_idx - 1], ((0, 0), (r_diff_1, r_diff_2), (c_diff_1, c_diff_2)), mode='symmetric')
                else:

                    self.new_dataset[mat_idx - 1, inst_idx - 1] = dataset[mat_idx - 1, inst_idx - 1]

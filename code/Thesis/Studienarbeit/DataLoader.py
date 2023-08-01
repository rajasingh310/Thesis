import os
import scipy
import pandas as pd
import openpyxl


class DataLoader:
    def __init__(self, dataset_dir='dataset_2'):
        self.data, self.mat_list_names, self.pix_ranges = self.load_tof_dataset(dataset_dir)

    def load_tof_dataset(self, dataset_dir):

        # Get the current script's directory
        current_dir = os.path.dirname(os.path.realpath(__file__))

        full_dataset_dir = os.path.join(current_dir, '..//..//..//datasets', dataset_dir)

        # Find .mat file in the directory
        mat_files = [f for f in os.listdir(full_dataset_dir) if f.endswith('.mat')]

        if len(mat_files) == 0:
            print('No .mat files found in directory')
        else:
            # Load the first .mat file found
            file_path = os.path.join(full_dataset_dir, mat_files[0])
            data = scipy.io.loadmat(file_path)
            #import h5py
            #data = h5py.File(file_path);

        dataset = data['dataset']

        # Find .xlsx file in the directory
        xlsx_files = [f for f in os.listdir(full_dataset_dir) if f.endswith('.xlsx')]

        if len(xlsx_files) == 0:
            print('No .xlsx files found in directory')
        else:
            # Load the first .xlsx file found
            file_path = os.path.join(full_dataset_dir, xlsx_files[0])
            mat_list_names = pd.read_excel(file_path, header=None, skiprows=1, nrows=dataset.shape[0], usecols='A')
            mat_list_names = mat_list_names.values

            # Load the second .xlsx file found
            file_path = os.path.join(full_dataset_dir, xlsx_files[1])
            # Load the Excel file
            workbook = openpyxl.load_workbook(file_path)
            sheet = workbook.active

            # Find the last column in row 3 (assuming header row)
            last_column = None
            for cell in sheet[3]:
                if cell.value is None:
                    last_column = cell.column - 1
                    break

            # Set the column range for loading data
            start_col = 'C'
            end_col = openpyxl.utils.get_column_letter(last_column)

            # Construct the range string
            pix_range_cols = f"{start_col}:{end_col}"

            # Read the data using pandas
            pix_ranges = pd.read_excel(file_path, header=None, skiprows=2, nrows=2 * dataset.shape[1], usecols=pix_range_cols)
            pix_ranges = pix_ranges.values

            return data, mat_list_names, pix_ranges

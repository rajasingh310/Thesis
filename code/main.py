# matlab to python

import os
import scipy
import pandas as pd
from normalization import normalization
from downSample import downsample_dataset
from find_sigma1 import find_sigma1
from images_labels import images_and_labels
import numpy as np
import math
from deep_neural_networks import algorithm_fcnn
from ConditionalGANs import ConditionalGANs

log_write = True
preload_data = True

if preload_data == False:
    '''
        Directory name for dataset:
        dataset_1 = Fifteen material (15M) dataset or 12M
        dataset_2 = Fourteen material (14M) dataset
        dataset_3 = Five material dataset acquired by Mr. Santosh Kasam in the year 2021

    '''

    # Directory path
    dataset_dir = 'dataset_2'

    full_dataset_dir = os.path.join('..\\datasets', dataset_dir)
    print(full_dataset_dir)

    # Find .mat file in the directory
    mat_files = [f for f in os.listdir(full_dataset_dir) if f.endswith('.mat')]

    if len(mat_files) == 0:
        print('No .mat files found in directory')
    else:
        # Load the first .mat file found
        file_path = os.path.join(full_dataset_dir, mat_files[0])
        data = scipy.io.loadmat(file_path)

    # List all the variable names present in the .mat file
    print(data.keys())

    dataset = data['dataset']
    amp_imgs = data['amp_imgs']
    depth_imgs = data['depth_imgs']
    f_0 = data['f_0']
    c = data['c']

    if dataset_dir == 'dataset_1':
        pass

    elif dataset_dir == 'dataset_2':
        mat_num = 14
        inst_num = 3
        pix_range_cols = 'C:AD'
        height_window = 60
        width_window = 80
    elif dataset_dir == "dataset_3":
        mat_num = 5
        inst_num = 70
        pix_range_cols = 'C:L'
        height_window = 60
        width_window = 80

    # Find .xlsx file in the directory
    xlsx_files = [f for f in os.listdir(full_dataset_dir) if f.endswith('.xlsx')]

    if len(xlsx_files) == 0:
        print('No .xlsx files found in directory')
    else:
        # Load the first .xlsx file found
        file_path = os.path.join(full_dataset_dir, xlsx_files[0])
        mat_list_names = pd.read_excel(file_path, header=None, skiprows=1, nrows=mat_num, usecols='A')
        mat_list_names = mat_list_names.values
        print(mat_list_names)
        # Load the second .xlsx file found
        file_path = os.path.join(full_dataset_dir, xlsx_files[1])
        pix_ranges = pd.read_excel(file_path, header=None, skiprows=2, nrows=2 * inst_num, usecols=pix_range_cols)
        pix_ranges = pix_ranges.values

    # Call the normalization function with the appropriate arguments
    new_dataset = normalization(dataset, amp_imgs, depth_imgs, f_0, c, pix_ranges)
    train_val_test = downsample_dataset(new_dataset, height_window, width_window)  # balancing the dataset
    sigma = find_sigma1(train_val_test)

    window_rows = 5
    window_cols = 5
    window_stride = 1
    k_nearest = 9

    [x_data, y_data] = images_and_labels(train_val_test, window_rows, window_cols, window_stride, mat_list_names, k_nearest, sigma)

    # Save the data
    np.savez('data.npz', x=x_data, y=y_data)

# Load the data
loaded_data = np.load('data.npz')
x_data = loaded_data['x']
y_data = loaded_data['y']




# Get the size of the dimension to be shuffled
dim_size = x_data.shape[0]

# Generate a random permutation of the indices along the dimension to be shuffled
perm = np.random.permutation(dim_size)

# Use advanced indexing to reorder the array according to the permutation
shuffled_x_data = x_data[perm, :, :, :, :]
shuffled_y_data = np.squeeze(y_data[perm, :], axis=None)


train_val_ratio = 0.7

x_train = shuffled_x_data[:math.floor(train_val_ratio*dim_size), :, :, :, :]
y_train = shuffled_y_data[:math.floor(train_val_ratio*dim_size), ]
x_val = shuffled_x_data[math.floor(train_val_ratio*dim_size+1):, :, :, :, :]
y_val = shuffled_y_data[math.floor(train_val_ratio*dim_size+1):, ]

# Execute algorithm_fcnn function
#net = algorithm_fcnn(x_train, x_val, y_train, y_val, log_write)

# Print final result
print("Training completed!")
num_samples = 200
generated_data, generated_data_labels = ConditionalGANs(x_data, y_data, num_samples, log_write)






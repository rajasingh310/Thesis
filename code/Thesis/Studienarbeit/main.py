import torch
import numpy as np
import math

from find_sigma1 import find_sigma1
from images_labels import images_and_labels
from deep_neural_networks import algorithm_fcnn

preload_data = True
crop_pixel_of_interest = True
pad = False
log_write = False
save_dataset = True
training = True
save_network = True

dataset_dir = 'dataset_2'

if not preload_data:
    '''
        Directory name for dataset:
        dataset_1 = Fifteen material (15M) dataset or 12M
        dataset_2 = Fourteen material (14M) dataset
        dataset_3 = Five material dataset acquired by Mr. Santosh Kasam in the year 2021

    '''

    from DataLoader import DataLoader
    load_data = DataLoader(dataset_dir=dataset_dir)

    # Data Normalization and if crop_pixel_of_interest is True, then do pixel cropping
    from DataNormalization import Normalization
    normalized_dataset = Normalization(load_data.data, load_data.pix_ranges, crop_pixel_of_interest)

    mat_list_names = load_data.mat_list_names
    new_dataset = normalized_dataset.new_dataset

    # Delete the specified row
    if dataset_dir == 'dataset_1':
        mat_list_names = np.delete(mat_list_names, [6, 10, 13], axis=0)
        new_dataset = np.delete(new_dataset, [6, 10, 13], axis=0)

    # Corrected the orientation of the data and then, if pad is True, then do padding
    from Data_Padding import Padding
    padded_dataset = Padding(new_dataset, load_data.pix_ranges, pad=pad)

    from DataFeatures import FeatureExtraction
    feature_dataset = FeatureExtraction(padded_dataset.new_dataset)

    sigma = find_sigma1(feature_dataset.r_i_new_dataset)
    np.savez('../../../results/' + str(dataset_dir) + '/sigma.npz', sigma=sigma)

    window_rows = window_cols = 5
    window_stride = 1
    k_nearest = 9

    [x_data, y_data] = images_and_labels(feature_dataset.r_i_new_dataset, window_rows, window_cols, window_stride, k_nearest, sigma, shuffle=True)

    if save_dataset:
        np.savez('sliding_window_' + dataset_dir + '.npz', labels=y_data, data=x_data, allow_pickle=True)

# Load the data
loaded_data = np.load('sliding_window_' + dataset_dir + '.npz')
x_data = loaded_data['data']
y_data = loaded_data['labels']

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
if training:
    net = algorithm_fcnn(x_train, x_val, y_train, y_val, log_write)

if save_network:
    torch.save(net.state_dict(), '../../../results/' + str(dataset_dir) + '/trained_model.pth')

from deep_neural_networks import Net
model = Net(input_shape=x_train.shape, output_shape=len(np.unique(y_train))).to('cpu')

trained_model = torch.load('../../../results/' + str(dataset_dir) + '/trained_model.pth')

model.load_state_dict(trained_model)
model.eval()

output0 = torch.argmax(model(torch.Tensor(shuffled_x_data)), 1)
accuracy0 = (output0 == torch.Tensor(shuffled_y_data)).float().mean().item() * 100

output1 = torch.argmax(model(torch.Tensor(x_train)), 1)
accuracy1 = (output1 == torch.Tensor(y_train)).float().mean().item() * 100

output2 = torch.argmax(model(torch.Tensor(x_val)), 1)
accuracy2 = (output2 == torch.Tensor(y_val)).float().mean().item() * 100

a = 2
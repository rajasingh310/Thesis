import numpy as np

preload_data = True
pad = True
crop_pixel_of_interest = True
save_images = False
filter = True
log_write = True

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

    dataset = load_data.data['dataset']    #Load the dataset
    mats = dataset.shape[0]    # Number of materials
    inst = dataset.shape[1]    # Number of Instances

    # Data Normalization and if crop_pixel_of_interest is True, then do pixel cropping
    from DataNormalization import Normalization
    normalized_dataset = Normalization(load_data.data, load_data.pix_ranges, crop_pixel_of_interest)

    # Corrected the orientation of the data and then, if pad is True, then do padding
    from Data_Padding import Padding
    padded_dataset = Padding(normalized_dataset.new_dataset, pad=pad)

    from feature_extraction import FeatureExtraction
    feature_dataset = FeatureExtraction(padded_dataset.new_dataset)

    from DataFiltering import DataFilter
    filtered_dataset = DataFilter(feature_dataset.abs_new_dataset, dataset_dir, save_images=save_images, filter=filter)

    from het_tof_image import HetToFImage
    het_dataset = HetToFImage(filtered_dataset.new_dataset, filtered_dataset.z_,  feature_dataset.r_i_new_dataset, n_samples=1000, h=172, w=224, mats=None)

    # from DataImbalance import HandleDataImbalance
    # balanced_dataset = HandleDataImbalance(normalized_dataset.new_dataset, dataBalance=False)

    #  Save the data
    if filter:
        np.savez('het_dataset_filter_' + dataset_dir + '.npz', labels=het_dataset.labels, abs_dataset=het_dataset.new_abs_dataset, r_i_dataset=het_dataset.new_r_i_dataset, num_mats=het_dataset.num_mats, allow_pickle=True)
    else:
        np.savez('het_dataset_' + dataset_dir + '.npz', labels=het_dataset.labels, abs_dataset=het_dataset.new_abs_dataset, r_i_dataset=het_dataset.new_r_i_dataset, num_mats=het_dataset.num_mats, allow_pickle=True)

# Load the data
if filter:
    loaded_data = np.load('het_dataset_filter_' + dataset_dir + '.npz', allow_pickle=True)
else:
    loaded_data = np.load('het_dataset_' + dataset_dir + '.npz', allow_pickle=True)

abs_data = loaded_data['abs_dataset']
r_i_dataset = loaded_data['r_i_dataset']
labels = loaded_data['labels']
num_mats = loaded_data['num_mats']

gen_input_data = np.concatenate(abs_data[:, 0], axis=0)
gen_input_data = gen_input_data[np.newaxis, np.newaxis, ...]
gen_input_data = np.reshape(gen_input_data, (-1, 1, 172, 224))

disc_input_data = np.concatenate(r_i_dataset[:, 0], axis=0)
disc_input_data = disc_input_data[np.newaxis, ...]
disc_input_data = np.reshape(disc_input_data, (-1, 2, 8, 172, 224))

labels = np.concatenate(labels[:, 0], axis=0)
labels = np.reshape(labels, (-1, 172, 224))

from GANs import GANs
GANs(gen_input_data, disc_input_data, labels, log_write=log_write)

a = 2

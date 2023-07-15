import numpy as np
import torch
import random
import sys

sys.path.append('Studienarbeit/')

import matplotlib
#matplotlib.use('Agg')

preload_data = True
pad = True
crop_pixel_of_interest = True
save_images = False
filter = False
log_write = False
xx = True
training = False
save_network = False

dataset_dir = 'dataset_2'

if not preload_data:
    '''
        Directory name for dataset:
        dataset_1 = Fifteen material (15M) dataset or 12M
        dataset_2 = Fourteen material (14M) dataset
        dataset_3 = Five material dataset acquired by Mr. Santosh Kasam in the year 2021

    '''

    from Studienarbeit.DataLoader import DataLoader
    load_data = DataLoader(dataset_dir=dataset_dir)

    dataset = load_data.data['dataset']    #Load the dataset
    mats = dataset.shape[0]    # Number of materials
    inst = dataset.shape[1]    # Number of Instances

    # Data Normalization and if crop_pixel_of_interest is True, then do pixel cropping
    from Studienarbeit.DataNormalization import Normalization
    normalized_dataset = Normalization(load_data.data, load_data.pix_ranges, crop_pixel_of_interest)

    # Corrected the orientation of the data and then, if pad is True, then do padding
    from Studienarbeit.Data_Padding import Padding
    padded_dataset = Padding(normalized_dataset.new_dataset, load_data.pix_ranges, pad=pad)

    from Studienarbeit.feature_extraction import FeatureExtraction
    feature_dataset = FeatureExtraction(padded_dataset.new_dataset)

    from DataFiltering import DataFilter
    filtered_dataset = DataFilter(feature_dataset.abs_new_dataset, dataset_dir, save_images=save_images, filter=filter)

    from het_tof_image import HetToFImage
    het_dataset = HetToFImage(filtered_dataset.new_dataset, filtered_dataset.z_,  feature_dataset.r_i_new_dataset, n_samples=1000, h=172, w=224, mats=None)

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

gen_input_data = np.concatenate(abs_data[:, 0], axis=0)
gen_input_data = gen_input_data[np.newaxis, np.newaxis, ...]
gen_input_data = np.reshape(gen_input_data, (-1, 1, 172, 224))

disc_input_data = np.concatenate(r_i_dataset[:, 0], axis=0)
disc_input_data = disc_input_data[np.newaxis, ...]
disc_input_data = np.reshape(disc_input_data, (-1, 2, 8, 172, 224))

gans_labels = np.concatenate(labels[:, 0], axis=0)
gans_labels = np.reshape(gans_labels, (-1, 172, 224))

if training:
    from GANs import GANs
    gen_network = GANs(gen_input_data[:300, ...], disc_input_data[:300, ...], gans_labels[:300, ...], loss='BCE', log_write=log_write)

    if save_network:
        torch.save(gen_network.gen.state_dict(), '../../results/' + str(dataset_dir) + '/gen_trained_model.pth')


if xx:

    for _ in range(10):

        j = np.random.randint(low=0, high=100)

        sigma = np.load('../../results/' + str(dataset_dir) + '/sigma.npz')
        sigma = sigma['sigma']

        from Studienarbeit.images_labels import images_and_labels
        [x_data, y_data] = images_and_labels(r_i_dataset[j:j+1, ...], 5, 5, 1, 9, sigma, shuffle=False)

        from Studienarbeit.deep_neural_networks import Net
        model1 = Net(input_shape=x_data.shape, output_shape=14).to('cpu')

        trained_model = torch.load('../../results/' + str(dataset_dir) + '/trained_model.pth')
        model1.load_state_dict(trained_model)
        model1.eval()

        output1 = torch.argmax(model1(torch.Tensor(x_data).to('cpu')), 1)
        accuracy1 = (output1 == torch.Tensor(np.ravel(gans_labels[j, ...]))).float().mean().item() * 100

        from GANs import Generator
        model2 = Generator(gen_input_data[:100, ...].shape).to('cuda')
        trained_model = torch.load('../../results/' + str(dataset_dir) + '/gen_trained_model.pth')
        model2.load_state_dict(trained_model)
        model2.eval()

        fake = model2(torch.from_numpy(gen_input_data[:100, ...]).to(torch.float).to('cuda'))
        fake = fake.detach().to('cpu').numpy()

        fake_imgs = np.empty((fake.shape[0], 1), dtype=object)
        for i in range(len(fake)):
            fake_imgs[i, 0] = fake[i, ...]

        [x_data, y_data] = images_and_labels(fake_imgs[j:j+1, ...], 5, 5, 1, 9, sigma, shuffle=False)
        output2 = torch.argmax(model1(torch.Tensor(x_data).to('cpu')), 1)
        accuracy2 = (output2 == torch.Tensor(np.ravel(gans_labels[j, ...]))).float().mean().item() * 100

        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        from matplotlib.colors import ListedColormap

        fig = plt.figure()

        norm = colors.Normalize(vmin=abs_data[j, 0].min(), vmax=abs_data[j, 0].max())
        fig.add_subplot(331).imshow(abs_data[j, 0], cmap='gray', norm=norm)
        norm = colors.Normalize(vmin=r_i_dataset[j, 0][1, 1, ...].min(), vmax=r_i_dataset[j, 0][1, 1, ...].max())
        fig.add_subplot(332).imshow(r_i_dataset[j, 0][1, 1, ...], cmap='gray', norm=norm)
        norm = colors.Normalize(vmin=fake_imgs[j, 0][1, 1, ...].min(), vmax=fake_imgs[j, 0][1, 1, ...].max())
        fig.add_subplot(333).imshow(fake_imgs[j, 0][1, 1, ...], cmap='gray', norm=norm)

        fig.add_subplot(334).imshow(gans_labels[j, ...])
        fig.add_subplot(335).imshow(torch.reshape(output1, (172, 224)).to('cpu'))
        fig.add_subplot(336).imshow(torch.reshape(output2, (172, 224)).to('cpu'))

        plt.show()
        # plt.savefig('a_' + str(npzu) + '.pdf', dpi=600, bbox_inches='tight')
        plt.close()

a = 2
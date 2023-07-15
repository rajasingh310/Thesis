import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


class DataFilter:
    def __init__(self, dataset, dataset_dir, save_images=False, filter=False):

        if save_images:

            directory = '../results/' + dataset_dir + '/filter'
            # Check if the directory already exists
            if not os.path.exists(directory):

                # Create the new directory
                os.makedirs(directory)
                print("New directory created.")

            for i in range(dataset.shape[0]):
                for j in range(dataset.shape[1]):

                    z = dataset[i, j]

                    # save the 2d image
                    fig, ax = plt.subplots()
                    ax.imshow(z, cmap='gray')
                    ax.axis('off')
                    plt.savefig('../results/' + dataset_dir + '/filter' + '/i_mat_' + str(i) + '__inst_' + str(j) + '.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
                    plt.close()

                    # Generate x and y indices based on the shape of the array
                    x_indices = np.arange(z.shape[0])
                    y_indices = np.arange(z.shape[1])
                    y, x = np.meshgrid(y_indices, x_indices)

                    # Create a 3D surface plot
                    fig = plt.figure()

                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot_surface(x, y, z, cmap='jet')
                    ax.set_zlim(np.min(z), np.max(z))
                    ax.set_xlabel('X', fontsize=20, labelpad=-13)
                    ax.set_ylabel('Y', fontsize=20, labelpad=-13)
                    ax.set_zlabel('Z', fontsize=20, labelpad=-13)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_zticklabels([])

                    plt.savefig('../results/' + dataset_dir + '/filter' + '/amplitude_mat_' + str(i) + '__inst_' + str(j) + '.pdf', dpi=600, bbox_inches='tight')
                    plt.close()

                    # Create a 3D surface plot
                    fig = plt.figure()

                    ax = fig.add_subplot(111, projection='3d')
                    ax.contour(x, y, z, zdir='z', offset=np.min(z), cmap='jet')
                    ax.contour(x, y, z, zdir='x', offset=0, cmap='jet')
                    ax.contour(x, y, z, zdir='y', offset=np.max(y), cmap='jet')
                    ax.set(xlim=(0, np.max(x)), ylim=(0, np.max(y)), zlim=(np.min(z), np.max(z)))
                    ax.set_xlabel('X', fontsize=20, labelpad=-13)
                    ax.set_ylabel('Y', fontsize=20, labelpad=-13)
                    ax.set_zlabel('Z', fontsize=20, labelpad=-13)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_zticklabels([])

                    plt.savefig('../results/' + dataset_dir + '/filter' + '/contour_mat_' + str(i) + '__inst_' + str(j) + '.pdf', dpi=600,
                                bbox_inches='tight')
                    plt.close()

                    sigma = list(range(1, 6))
                    sigma = [x * 3 for x in sigma]
                    for k in range(len(sigma)):

                        z_ = gaussian_filter(z, sigma=sigma[k])

                        # save the 2d image
                        fig, ax = plt.subplots()
                        ax.imshow(z_, cmap='gray')
                        ax.axis('off')
                        plt.savefig('../results/' + dataset_dir + '/filter' + '/i_mat_' + str(i) + '__inst_' + str(j) + '__sigma_' + str(sigma[k]) + '.pdf', dpi=600,
                                    bbox_inches='tight', pad_inches=0)
                        plt.close()

                        # Create a 3D surface plot
                        fig = plt.figure()

                        ax = fig.add_subplot(111, projection='3d')
                        ax.plot_surface(x, y, z_, cmap='jet')
                        ax.set_zlim(np.min(z), np.max(z))
                        ax.set_xlabel('X', fontsize=20, labelpad=-13)
                        ax.set_ylabel('Y', fontsize=20, labelpad=-13)
                        ax.set_zlabel('Z', fontsize=20, labelpad=-13)
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_zticklabels([])

                        plt.savefig('../results/' + dataset_dir + '/filter' + '/amplitude_mat_' + str(i) + '__inst_' + str(j) + '__sigma_' + str(sigma[k])  + '.pdf', dpi=600,
                                    bbox_inches='tight')
                        plt.close()

                        # Create a 3D surface plot
                        fig = plt.figure()

                        ax = fig.add_subplot(111, projection='3d')
                        ax.contour(x, y, z_, zdir='z', offset=np.min(z), cmap='jet')
                        ax.contour(x, y, z_, zdir='x', offset=0, cmap='jet')
                        ax.contour(x, y, z_, zdir='y', offset=np.max(y), cmap='jet')
                        ax.set(xlim=(0, np.max(x)), ylim=(0, np.max(y)), zlim=(np.min(z), np.max(z)))
                        ax.set_xlabel('X', fontsize=20, labelpad=-13)
                        ax.set_ylabel('Y', fontsize=20, labelpad=-13)
                        ax.set_zlabel('Z', fontsize=20, labelpad=-13)
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_zticklabels([])

                        plt.savefig('../results/' + dataset_dir + '/filter' + '/contour_mat_' + str(i) + '__inst_' + str(j) + '__sigma_' + str(sigma[k]) + '.pdf', dpi=600,
                                    bbox_inches='tight')
                        plt.close()

        self.z_ = np.empty((1, dataset.shape[1]), dtype=object)
        if filter:

            for j in range(dataset.shape[1]):
                if dataset_dir == 'dataset_1':
                    self.z_[0, j] = dataset[1, j] / gaussian_filter(dataset[1, j], sigma=6)

                elif dataset_dir == 'dataset_2':
                    self.z_[0, j] = dataset[8, j] / gaussian_filter(dataset[8, j], sigma=6)

                elif dataset_dir == 'dataset_3':
                    self.z_[0, j] = dataset[3, j] / gaussian_filter(dataset[3, j], sigma=6)

            self.new_dataset = dataset / self.z_

        else:
            self.new_dataset = dataset




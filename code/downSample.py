# Import necessary packages
import numpy as np


def downsample_dataset(data, height, width):
    """
    This function returns a random crop of the image according to the size prescribed.

    Args:
    - data (ndarray): input data of shape (num_samples, num_features, num_rows, num_cols).
    - height (int): height of the output image.
    - width (int): width of the output image.

    Returns:
    - downsampled_dataset (ndarray): output data of shape (num_samples, num_features, height, width).
    """

    # Create a new empty array to hold the output
    downsampled_dataset = np.empty((data.shape[0], data.shape[1]), dtype=object)

    for i in range(data.shape[0]):  # Loop through each sample
        for j in range(data.shape[1]):  # Loop through each feature

            # Extract the real and imaginary components of the data
            real_value = np.real(data[i, j])
            imag_value = np.imag(data[i, j])

            # Stack the real and imaginary components along the channel axis
            new_data = np.stack([real_value, imag_value], axis=-1)

            # Determine the input and target shapes
            input_shape = new_data.shape

            # randomly select the starting position of the crop
            start_row = np.random.randint(0, input_shape[0]-height)
            b = input_shape[1] - width
            if b == 0:
                start_col = 0
            else:
                start_col = np.random.randint(0, b)

            # perform the crop operation
            cropped_image = new_data[start_row:start_row + height, start_col:start_col + width, :, :]
            # Save the cropped image to the output array
            downsampled_dataset[i, j] = cropped_image

    return downsampled_dataset

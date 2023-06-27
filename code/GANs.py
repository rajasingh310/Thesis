def GANs(real_data, real_labels, num_samples, log_write):

    import torch
    from torch import nn
    from tqdm.auto import tqdm
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    import numpy as np
    from tensorboardX import SummaryWriter

    def get_generator_block(input_dim, output_dim):
        '''
        Function for returning a block of the generator's neural network
        given input and output dimensions.
        Parameters:
            input_dim: the dimension of the input vector, a scalar
            output_dim: the dimension of the output vector, a scalar
        Returns:
            a generator neural network layer, with a linear transformation
              followed by a batch normalization and then a relu activation
        '''
        return nn.Sequential(
            # Hint: Replace all of the "None" with the appropriate dimensions.
            # The documentation may be useful if you're less familiar with PyTorch:
            # https://pytorch.org/docs/stable/nn.html.
            #### START CODE HERE ####
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            #### END CODE HERE ####
            nn.ReLU(inplace=True)
        )

    class Generator(nn.Module):
        '''
        Generator Class
        Values:
            z_dim: the dimension of the noise vector, a scalar
            im_dim: the dimension of the images, fitted for the dataset used, a scalar
              (MNIST images are 28 x 28 = 784 so that is your default)
            hidden_dim: the inner dimension, a scalar
        '''

        def __init__(self, input_dim=10, output_dim=784, hidden_dim=32):
            super(Generator, self).__init__()
            # Build the neural network
            self.gen = nn.Sequential(
                get_generator_block(input_dim, hidden_dim),
                get_generator_block(hidden_dim, hidden_dim * 2),
                get_generator_block(hidden_dim * 2, hidden_dim * 4),
                get_generator_block(hidden_dim * 4, hidden_dim * 8),
                get_generator_block(hidden_dim * 8, hidden_dim * 16),
                get_generator_block(hidden_dim * 16, hidden_dim * 32),
                get_generator_block(hidden_dim * 32, hidden_dim * 64),
                get_generator_block(hidden_dim * 64, hidden_dim * 128),
                get_generator_block(hidden_dim * 128, hidden_dim * 64),
                # There is a dropdown with hints if you need them!
                #### START CODE HERE ####
                nn.Linear(hidden_dim * 64, output_dim),
                nn.Sigmoid()
                #### END CODE HERE ####
            )

        def forward(self, noise):
            '''
            Function for completing a forward pass of the generator: Given a noise tensor,
            returns generated images.
            Parameters:
                noise: a noise tensor with dimensions (n_samples, z_dim)
            '''
            return self.gen(noise)

    def get_noise(n_samples, input_dim, device='cpu'):
        '''
        Function for creating noise vectors: Given the dimensions (n_samples, input_dim)
        creates a tensor of that shape filled with random numbers from the normal distribution.
        Parameters:
            n_samples: the number of samples to generate, a scalar
            input_dim: the dimension of the input vector, a scalar
            device: the device type
        '''
        return torch.randn(n_samples, input_dim, device=device)

    def get_discriminator_block(input_dim, output_dim):
        '''
        Discriminator Block
        Function for returning a neural network of the discriminator given input and output dimensions.
        Parameters:
            input_dim: the dimension of the input vector, a scalar
            output_dim: the dimension of the output vector, a scalar
        Returns:
            a discriminator neural network layer, with a linear transformation
              followed by an nn.LeakyReLU activation with negative slope of 0.2
              (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html)
        '''
        return nn.Sequential(
            #### START CODE HERE ####
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2)
            #### END CODE HERE ####
        )

    class Discriminator(nn.Module):
        '''
        Discriminator Class
        Values:
            im_dim: the dimension of the images, fitted for the dataset used, a scalar
                (MNIST images are 28x28 = 784 so that is your default)
            hidden_dim: the inner dimension, a scalar
        '''

        def __init__(self, input_dim=784, output_dim=1, hidden_dim=32):
            super(Discriminator, self).__init__()
            self.disc = nn.Sequential(
                get_discriminator_block(input_dim, hidden_dim * 4),
                get_discriminator_block(hidden_dim * 4, hidden_dim * 8),
                get_discriminator_block(hidden_dim * 8, hidden_dim * 16),
                get_discriminator_block(hidden_dim * 16, hidden_dim * 32),
                get_discriminator_block(hidden_dim * 32, hidden_dim * 16),
                get_discriminator_block(hidden_dim * 16, hidden_dim * 8),
                get_discriminator_block(hidden_dim * 8, hidden_dim * 4),
                get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
                get_discriminator_block(hidden_dim * 2, hidden_dim),
                # Hint: You want to transform the final output into a single value,
                #       so add one more linear map.
                #### START CODE HERE ####
                nn.Linear(hidden_dim, output_dim)
                #### END CODE HERE ####
            )

        def forward(self, image):
            '''
            Function for completing a forward pass of the discriminator: Given an image tensor,
            returns a 1-dimension tensor representing fake/real.
            Parameters:
                image: a flattened image tensor with dimension (im_dim)
            '''
            return self.disc(image)

    def get_one_hot_labels(labels, n_classes):
        '''
        Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes).
        Parameters:
            labels: tensor of labels from the dataloader, size (?)
            n_classes: the total number of classes in the dataset, an integer scalar
        '''

        return F.one_hot(labels, n_classes)

    assert (
            get_one_hot_labels(
                labels=torch.Tensor([[0, 2, 1]]).long(),
                n_classes=3
            ).tolist() ==
            [[
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0]
            ]]
    )


    # Check that the device of get_one_hot_labels matches the input device
    if torch.cuda.is_available():
        assert str(get_one_hot_labels(torch.Tensor([[0]]).long().cuda(), 1).device).startswith("cuda")

    print("Success!")

    def combine_vectors(x, y):
        '''
        Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).
        Parameters:
          x: (n_samples, ?) the first vector.
            In this assignment, this will be the noise vector of shape (n_samples, z_dim),
            but you shouldn't need to know the second dimension's size.
          y: (n_samples, ?) the second vector.
            Once again, in this assignment this will be the one-hot class vector
            with the shape (n_samples, n_classes), but you shouldn't assume this in your code.
        '''
        # Note: Make sure this function outputs a float no matter what inputs it receives

        combined = torch.cat((x.float(), y.float()), dim=1)
        return combined

    combined = combine_vectors(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]]))
    if torch.cuda.is_available():
        # Check that it doesn't break with cuda
        cuda_check = combine_vectors(torch.tensor([[1, 2], [3, 4]]).cuda(), torch.tensor([[5, 6], [7, 8]]).cuda())
        assert str(cuda_check.device).startswith("cuda")
    # Check exact order of elements
    assert torch.all(combined == torch.tensor([[1, 2, 5, 6], [3, 4, 7, 8]]))
    # Tests that items are of float type
    assert (type(combined[0][0].item()) == float)
    # Check shapes
    combined = combine_vectors(torch.randn(1, 4, 5), torch.randn(1, 8, 5))
    assert tuple(combined.shape) == (1, 12, 5)
    assert tuple(combine_vectors(torch.randn(1, 10, 12).long(), torch.randn(1, 20, 12).long()).shape) == (1, 30, 12)
    # Check that the float transformation doesn't happen after the inputs are concatenated
    assert tuple(combine_vectors(torch.randn(1, 10, 12).long(), torch.randn(1, 20, 12)).shape) == (1, 30, 12)
    print("Success!")

    # ## Training
    # Now you can start to put it all together!
    # First, you will define some new parameters:
    #
    # *   n_classes: the number of classes in MNIST (10, since there are the digits from 0 to 9)

    # In[ ]:

    real_data_shape = (real_data.shape[1], real_data.shape[2], real_data.shape[3], real_data.shape[4])
    n_classes = len(np.unique(real_labels))

    # And you also include the same parameters from previous assignments:
    #
    #   *   criterion: the loss function
    #   *   n_epochs: the number of times you iterate through the entire dataset when training
    #   *   z_dim: the dimension of the noise vector
    #   *   display_step: how often to display/visualize the images
    #   *   batch_size: the number of images per forward/backward pass
    #   *   lr: the learning rate
    #   *   device: the device type
    #

    # Define the number of iterations for critic and generator updates
    n_critic = 5
    n_generator = 1

    # Define a function to compute Wasserstein loss
    def wasserstein_loss(real_score, fake_score):
        return -torch.mean(real_score) + torch.mean(fake_score)

    n_epochs = 20
    z_dim = 64
    batch_size = 1000
    lr = 0.0008
    device = 'cuda'

    real_data = np.reshape(real_data, (real_data.shape[0], 144))
    real_dataset = TensorDataset(torch.Tensor(real_data).to(device), torch.Tensor(real_labels).to(device))
    dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True)

    # Then, you can initialize your generator, discriminator, and optimizers. To do this, you will need to update the input dimensions for both models. For the generator, you will need to calculate the size of the input vector; recall that for conditional GANs, the generator's input is the noise vector concatenated with the class vector. For the discriminator, you need to add a channel for every class.

    # In[ ]:

    # UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
    # GRADED FUNCTION: get_input_dimensions
    def get_input_dimensions(z_dim, real_data_shape, n_classes):
        '''
        Function for getting the size of the conditional input dimensions
        from z_dim, the image shape, and number of classes.
        Parameters:
            z_dim: the dimension of the noise vector, a scalar
            mnist_shape: the shape of each MNIST image as (C, W, H), which is (1, 28, 28)
            n_classes: the total number of classes in the dataset, an integer scalar
                    (10 for MNIST)
        Returns:
            generator_input_dim: the input dimensionality of the conditional generator,
                              which takes the noise and class vectors
            discriminator_im_chan: the number of input channels to the discriminator
                                (e.g. C x 28 x 28 for MNIST)
        '''
        generator_input_dim = z_dim
        discriminator_im_chan = real_data_shape[0]*real_data_shape[1]*real_data_shape[2]*real_data_shape[3]
        return generator_input_dim, discriminator_im_chan

    generator_input_dim, discriminator_im_chan = get_input_dimensions(z_dim, real_data_shape, n_classes)

    gen = Generator(input_dim=generator_input_dim, output_dim=discriminator_im_chan).to(device)
    disc = Discriminator(input_dim=discriminator_im_chan).to(device)

    # Define the optimizer for critic and generator
    disc_opt = torch.optim.RMSprop(disc.parameters(), lr=lr)
    gen_opt = torch.optim.RMSprop(gen.parameters(), lr=lr)

    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    if log_write:
        writer = SummaryWriter()

    for epoch in range(n_epochs):
        # Dataloader returns the batches and the labels
        for real, labels in tqdm(dataloader):
            cur_batch_size = len(real)
            # Flatten the batch of real images from the dataset
            real = real.to(device)
            labels = torch.squeeze(labels.to(device)).long()

            # Update the discriminator for n_discriminator iterations
            for _ in range(n_critic):

                disc_opt.zero_grad()
                fake_noise = get_noise(cur_batch_size, z_dim, device=device)

                fake = gen(fake_noise)

                disc_fake_pred = disc(fake.detach())
                disc_real_pred = disc(real)


                # Compute the Wasserstein loss for critic
                disc_loss = wasserstein_loss(disc_real_pred, disc_fake_pred)
                disc_loss.backward(retain_graph=True)

                disc_opt.step()

                # Clip the critic weights to enforce Lipschitz constraint
                for p in disc.parameters():
                    p.data.clamp_(-0.01, 0.01)

            # Update the generator for n_generator iterations
            for _ in range(n_generator):

                gen_opt.zero_grad()

                fake = gen(fake_noise)

                disc_fake_pred = disc(fake)

                # Compute the Wasserstein loss for generator
                # Note that we use -fake_score because we want to maximize it
                gen_loss = -torch.mean(disc_fake_pred)
                gen_loss.backward()
                gen_opt.step()

        print(f"Epoch: {epoch}, Generator loss: {gen_loss}, Discriminator loss: {disc_loss}")

        if log_write:
            writer.add_scalar('Generator loss', gen_loss, epoch)
            writer.add_scalar('Discriminator loss', disc_loss, epoch)

    return 0, 0
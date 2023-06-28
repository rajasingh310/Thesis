import torch
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter


def g2d(input_channel, output_channel, kernel=5):
    return nn.Sequential(
        nn.BatchNorm2d(input_channel),
        nn.Conv2d(input_channel, output_channel, kernel),
        nn.ReLU(),
    )


def g3d(input_channel, output_channel, kernel=(5, 5, 5)):
    return nn.Sequential(
        nn.BatchNorm3d(input_channel),
        nn.ConvTranspose3d(input_channel, output_channel, kernel),
        nn.ReLU()
    )


class Generator(nn.Module):
    def __init__(self, gen_input_shape):
        super(Generator, self).__init__()

        self.gen_input_shape = gen_input_shape

        self.G2d_Net = nn.Sequential(
            g2d(self.gen_input_shape[1], self.gen_input_shape[1] * 2, kernel=5),
            g2d(self.gen_input_shape[1] * 2, self.gen_input_shape[1] * 4, kernel=5),
            g2d(self.gen_input_shape[1] * 4, self.gen_input_shape[1] * 8, kernel=5),
            g2d(self.gen_input_shape[1] * 8, self.gen_input_shape[1] * 8, kernel=5)
        )

        self.G3d_Net = nn.Sequential(
            g3d(self.gen_input_shape[1], self.gen_input_shape[1] * 2, kernel=(1, 10, 10)),
            g3d(self.gen_input_shape[1] * 2, self.gen_input_shape[1] * 2, kernel=(1, 8, 8))
        )

    def forward(self, gen_input):
        data = self.G2d_Net(gen_input)
        data = torch.unsqueeze(data, 1)
        data = self.G3d_Net(data)
        return data


def d3d(input_channel, output_channel, kernel=(5, 5, 5)):
    return nn.Sequential(
        nn.BatchNorm3d(input_channel),
        nn.Conv3d(input_channel, output_channel, kernel),
        nn.ReLU()
    )


class Discriminator(nn.Module):

    def __init__(self, disc_input_shape):
        super(Discriminator, self).__init__()

        self.disc_input_shape = disc_input_shape

        self.D_Net = nn.Sequential(
            d3d(self.disc_input_shape[1], self.disc_input_shape[1] * 4, kernel=(4, 10, 10)),
            d3d(self.disc_input_shape[1] * 4, self.disc_input_shape[1] * 8, kernel=(4, 10, 10)),
            d3d(self.disc_input_shape[1] * 8, self.disc_input_shape[1] * 32, kernel=(1, 10, 10)),
        )

    def forward(self, disc_input):
        data = self.D_Net(disc_input)
        layer = nn.Linear(data.shape[1] * data.shape[2] * data.shape[3] * data.shape[4], 1, device='cuda')
        data = torch.reshape(data, (data.shape[0], -1))
        return layer(data)


class GANs:

    def __init__(self, gen_input, disc_input, labels, log_write):
        self.gen_input = torch.from_numpy(gen_input).to(torch.float).to('cuda')
        self.disc_input = torch.from_numpy(disc_input).to(torch.float).to('cuda')
        self.labels = torch.from_numpy(labels)
        self.log_write = log_write

        self.gen_input_shape = self.gen_input.shape
        self.disc_input_shape = self.disc_input.shape

        Gen = Generator(self.gen_input_shape).to('cuda')
        fake = Gen(self.gen_input[:5, ...])

        Disc = Discriminator(self.disc_input_shape).to('cuda')
        y_pred = Disc(fake)

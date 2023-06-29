import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter
import time


def g2d(input_channel, output_channel, kernel=5):
    return nn.Sequential(
        nn.BatchNorm2d(input_channel),
        nn.Conv2d(input_channel, output_channel, kernel),
        nn.ReLU()
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


def wasserstein_loss(fake_score, real_score):
    return torch.mean(fake_score) - torch.mean(real_score)
class GANs:

    def __init__(self, gen_input, disc_input, labels, log_write, loss='BCE'):
        self.gen_input = torch.from_numpy(gen_input).to(torch.float).to('cuda')
        self.disc_input = torch.from_numpy(disc_input).to(torch.float).to('cuda')
        self.labels = torch.from_numpy(labels)
        self.log_write = log_write

        self.gen_input_shape = self.gen_input.shape
        self.disc_input_shape = self.disc_input.shape

        self.lr = 0.008
        self.n_epochs = 30
        self.batch_size = 3
        self.iter = 0
        self.clip_value = 0.01

        self.criterion = nn.BCEWithLogitsLoss()

        gen = Generator(self.gen_input_shape).to('cuda')
        disc = Discriminator(self.disc_input_shape).to('cuda')

        gen_opt = torch.optim.Adam(gen.parameters(), lr=self.lr)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=self.lr)

        dataset = TensorDataset(self.gen_input, self.disc_input, self.labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if log_write:
            writer = SummaryWriter()

        self.t0 = time.time()
        for epoch in range(self.n_epochs):
            self.iteration = 0
            for gen_batch_input, disc_batch_input, labels in dataloader:

                for _ in range(5):
                    disc_opt.zero_grad()
                    fake = gen(gen_batch_input)

                    disc_fake_pred = disc(fake.detach())
                    disc_real_pred = disc(disc_batch_input)

                    if loss == 'BEC':
                        disc_fake_loss = self.criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                        disc_real_loss = self.criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                        disc_loss = (disc_fake_loss + disc_real_loss) / 2
                    elif loss == 'wasserstein':
                        disc_loss = wasserstein_loss(disc_fake_pred, disc_real_pred)

                    disc_loss.backward(retain_graph=True)
                    disc_opt.step()

                    if loss == 'wasserstein':
                        for p in disc.parameters():
                            p.data.clamp_(-self.clip_value, self.clip_value)

                gen_opt.zero_grad()

                disc_fake_pred_for_gen = disc(fake)
                if loss == 'BCE':
                    gen_loss = self.criterion(disc_fake_pred_for_gen, torch.ones_like(disc_fake_pred))
                elif loss == 'wasserstein':
                    gen_loss = - torch.mean(disc_fake_pred_for_gen)

                gen_loss.backward()
                gen_opt.step()

                self.iteration += 1
                self.iter += 1
                self.current_t = time.time()

                print(f"Epoch: {epoch}, Iteration: {self.iteration}, Time: {int(self.current_t - self.t0)} sec., Generator loss: {gen_loss}, discriminator loss: {disc_loss}")

                if log_write:
                    writer.add_scalar('Generator loss', gen_loss, self.iter)
                    writer.add_scalar('Discriminator loss', disc_loss, self.iter)

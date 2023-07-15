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

        self.f_layers = nn.Sequential(
            nn.Linear(64 * 2 * 145 * 197, 1),
            #nn.ReLU(),
            #nn.BatchNorm1d(400),
            #nn.Linear(400, 1),
            #nn.ReLU(),
            #nn.BatchNorm1d(128),
            #nn.Linear(128, 1),
            #nn.ReLU(),
            #nn.BatchNorm1d(64),
            #nn.Linear(64, 1)
        )

    def forward(self, disc_input):
        data = self.D_Net(disc_input)
        data = data.view(data.shape[0], -1)    # 64 2 145 197
        data = self.f_layers(data)
        return data


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
        self.n_epochs = 20
        self.batch_size = 3
        self.iter = 0
        self.clip_value = 0.01

        self.criterion = nn.BCEWithLogitsLoss()

        self.gen = Generator(self.gen_input_shape).to('cuda')
        disc = Discriminator(self.disc_input_shape).to('cuda')

        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=self.lr)

        dataset = TensorDataset(self.gen_input, self.disc_input, self.labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if log_write:
            writer = SummaryWriter()

        self.t0 = time.time()
        for epoch in range(self.n_epochs):
            self.iteration = 0
            for gen_batch_input, disc_batch_input, labels in dataloader:

                for _ in range(1):
                    disc_opt.zero_grad()
                    fake = self.gen(gen_batch_input)

                    disc_fake = disc(fake.detach())
                    disc_real = disc(disc_batch_input)





                    if loss == 'BCE':
                        disc_fake_accuracy = ((disc_fake > 0.5).float() == torch.zeros_like(disc_fake)).float().mean().item()
                        disc_real_accuracy = ((disc_real > 0.5).float() == torch.ones_like(disc_fake)).float().mean().item()
                        disc_fake_loss = self.criterion(disc_fake, torch.zeros_like(disc_fake))
                        disc_real_loss = self.criterion(disc_real, torch.ones_like(disc_real))
                        disc_loss = (disc_fake_loss + disc_real_loss) / 2
                    elif loss == 'wasserstein':
                        disc_fake_accuracy = ((disc_fake < 0).float() == torch.zeros_like(disc_fake)).float().mean().item()
                        disc_real_accuracy = ((disc_real > 0).float() == torch.ones_like(disc_fake)).float().mean().item()
                        disc_fake_loss = torch.mean(disc_fake)
                        disc_real_loss = torch.mean(disc_real)
                        disc_loss = - wasserstein_loss(disc_fake, disc_real)

                    disc_loss.backward(retain_graph=True)
                    disc_opt.step()

                    if loss == 'wasserstein':
                        for p in disc.parameters():
                            p.data.clamp_(-self.clip_value, self.clip_value)

                gen_opt.zero_grad()

                disc_fake_for_gen = disc(fake)
                if loss == 'BCE':
                    gen_loss = self.criterion(disc_fake_for_gen, torch.ones_like(disc_fake))
                elif loss == 'wasserstein':
                    gen_loss = - torch.mean(disc_fake_for_gen)

                gen_loss.backward()
                gen_opt.step()

                self.iteration += 1
                self.iter += 1
                self.current_t = time.time()

                print(f"E: {epoch}, I: {self.iteration}, T: {int(self.current_t - self.t0)} sec., G_l: {abs(gen_loss)}, D_l: {abs(disc_loss)}, D_lf: {disc_fake_loss}, D_lr: {disc_real_loss}, D_fa: {disc_fake_accuracy}, D_ra: {disc_real_accuracy}")

                if log_write:
                    writer.add_scalar('Generator loss', abs(gen_loss), self.iter)
                    writer.add_scalar('Discriminator loss', abs(disc_loss), self.iter)
                    writer.add_scalar('Discriminator loss fake', abs(disc_fake_loss), self.iter)
                    writer.add_scalar('Discriminator loss real', abs(disc_real_loss), self.iter)
                    writer.add_scalar('Discriminator accuracy fake', abs(disc_fake_accuracy), self.iter)
                    writer.add_scalar('Discriminator accuracy real', abs(disc_real_accuracy), self.iter)

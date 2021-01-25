import torch.nn as nn
import torch.nn.functional as F

class CycleGAN(object):
    def __init__(self, state_space, latent_space):
        self.gen_1 = Generator(state_space, latent_space)
        self.gen_2 = Generator(state_space, latent_space)
        self.dis_1 = Discriminator(state_space)
        self.dis_2 = Discriminator(state_space)

    def train(self, x):



class Encoder(nn.Module):
    def __init__(self, input_size, latent_space):
        super().__init__()
        self.enc_1 = nn.Linear(input_size, 64)
        self.enc_2 = nn.Linear(64, 64)
        self.enc_3 = nn.Linear(64, latent_space)

    def forward(self, input_data):
        x = self.enc_1(input_data)
        x = self.enc_2(x)
        x = self.enc_3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_space, output_size):
        super().__init__()
        self.dec_1 = nn.Linear(latent_space, 64)
        self.dec_2 = nn.Linear(64, 64)
        self.dec_3 = nn.Linear(64, output_size)

    def forward(self, input_data):
        x = self.dec_1(input_data)
        x = self.dec_2(x)
        x = self.dec_3(x)
        return x


class Generator(nn.Module):
    def __init__(self, state_space, latent_space):
        super().__init__()
        self.encoder = Encoder(state_space, latent_space)
        self.decoder = Decoder(latent_space, state_space)
        self.latent_vector = None

    def forward(self, input_data):
        self.latent_vector = self.encoder(input_data)
        output = self.decoder(self.latent_vector)
        return output

class Discriminator(nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.dis_1 = nn.Linear(state_space, 64)
        self.dis_2 = nn.Linear(64, 64)
        self.dis_3 = nn.Linear(64, 2)

    def forward(self, input_data):
        x = self.dis_1(input_data)
        x = self.dis_2(x)
        x = F.softmax(self.dis_3(x))
        return x
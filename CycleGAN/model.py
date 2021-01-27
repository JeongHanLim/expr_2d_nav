import torch
import torch.nn as nn
import torch.nn.functional as F

class CycleGAN(nn.Module):
    def __init__(self, state_space, latent_space):
        super().__init__()
        self.gen_1 = Generator(latent_space)
        self.gen_2 = Generator(latent_space)
        self.dis_1 = Discriminator(latent_space)
        self.dis_2 = Discriminator(latent_space)
        self.enc_1 = Encoder(state_space, latent_space)
        self.enc_2 = Encoder(state_space, latent_space)
        self.dec_1 = Decoder(state_space, latent_space)
        self.dec_2 = Decoder(state_space, latent_space)
        self.latent_vector_1 = None
        self.latent_vector_2 = None
        self.transfer_latent_vector_1 = None
        self.transfer_latent_vector_2 = None
        self.cycle_latent_vector_1 = None
        self.cycle_latent_vector_2 = None
        self.cycle_state_1 = None
        self.cycle_state_2 = None
        self.decode_state_1 = None
        self.decode_state_2 = None
        self.discrim_1 = None
        self.discrim_2 = None

    def forward(self, state_1, state_2):
        self.latent_vector_1 = self.enc_1(state_1)
        self.latent_vector_2 = self.enc_2(state_2)

        self.transfer_latent_vector_2 = self.gen_1(self.latent_vector_1)
        self.cycle_latent_vector_1 = self.gen_2(self.transfer_latent_vector_2)
        self.transfer_latent_vector_1 = self.gen_2(self.latent_vector_2)
        self.cycle_latent_vector_2 = self.gen_1(self.transfer_latent_vector_1)

        self.decode_state_1 = self.dec_1(self.latent_vector_1)
        self.cycle_state_1 = self.dec_1(self.cycle_latent_vector_1)
        self.decode_state_2 = self.dec_2(self.latent_vector_2)
        self.cycle_state_2 = self.dec_2(self.cycle_latent_vector_2)
        self.discrim_1 = self.dis_1(self.transfer_latent_vector_1)
        self.discrim_2 = self.dis_2(self.transfer_latent_vector_2)

        return self.decode_state_1, self.decode_state_2, self.cycle_latent_vector_1, self.cycle_latent_vector_2, \
               self.latent_vector_1, self.latent_vector_2, self.discrim_1, self.discrim_2


class CycleGANGen(nn.Module):
    def __init__(self, state_space, latent_space):
        super().__init__()
        self.gen_1 = Generator(latent_space)
        self.gen_2 = Generator(latent_space)
        self.enc_1 = Encoder(state_space, latent_space)
        self.enc_2 = Encoder(state_space, latent_space)

    def forward(self, state_1, state_2):
        latent_vector_1 = self.enc_1(state_1)
        latent_vector_2 = self.enc_2(state_2)

        transfer_latent_vector_2 = self.gen_1(latent_vector_1)
        transfer_latent_vector_1 = self.gen_2(latent_vector_2)

        return transfer_latent_vector_1, transfer_latent_vector_2, latent_vector_1, latent_vector_2


class Encoder(nn.Module):
    def __init__(self, input_size, latent_space):
        super().__init__()
        self.enc_1 = nn.Linear(input_size, 64)
        self.enc_2 = nn.Linear(64, 64)
        self.z_mu = nn.Linear(64, latent_space)
        self.z_sig = nn.Linear(64, latent_space)

    def forward(self, input_data):
        x = self.enc_1(input_data)
        x = self.enc_2(x)
        mu = self.z_mu(x)
        sig = self.z_sig(x)
        normal_distrib = torch.distributions.normal.Normal(mu, sig)
        return normal_distrib.sample()

class Decoder(nn.Module):
    def __init__(self, output_size, latent_space):
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
    def __init__(self, latent_space):
        super().__init__()
        self.layer_1_2 = nn.Linear(latent_space//2, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, latent_space)

    def forward(self, x):
        batch, len = x.shape
        split_1 = x[:, :len//2]
        split_2 = x[:, len//2:]
        x = self.layer_1(split_2)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = torch.cat((split_1, x), 1)
        return x

class Discriminator(nn.Module):
    def __init__(self, latent_space):
        super().__init__()
        self.dis_1 = nn.Linear(latent_space, 64)
        self.dis_2 = nn.Linear(64, 64)
        self.dis_3 = nn.Linear(64, 1)

    def forward(self, input_data):
        x = self.dis_1(input_data)
        x = self.dis_2(x)
        x = self.dis_3(x)
        return x
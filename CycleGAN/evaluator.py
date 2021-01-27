import numpy as np


class Evaluator(object):
    def __init__(self):
        self.num_class = num_class
        self.latent_vector_1 = None
        self.latent_vector_2 = None
        self.decode_state_1 = None
        self.decode_state_2 = None
        self.cycle_latent_vector_1 = None
        self.cycle_latent_vector_2 = None
        self.input_state_1 = None
        self.input_state_2 = None

    def add_batch(self, input_state_1, input_state_2, latent_vector_1, latent_vector_2,
                  decode_state_1, decode_state_2, cycle_latent_vector_1, cycle_latent_vector_2):
        self.input_state_1 = input_state_1
        self.input_state_2 = input_state_2
        self.latent_vector_1 = latent_vector_1
        self.latent_vector_2 = latent_vector_2
        self.decode_state_1 = decode_state_1
        self.decode_state_2 = decode_state_2
        self.cycle_latent_vector_1 = cycle_latent_vector_1
        self.cycle_latent_vector_2 = cycle_latent_vector_2

    def enc_dec_loss_1(self):
        return np.square(self.latent_vector_1 - self.decode_state_1).mean()

    def enc_dec_loss_2(self):
        return np.square(self.latent_vector_2 - self.decode_state_2).mean()

    def cycle_loss_1(self):
        return np.square(self.latent_vector_1 - self.cycle_latent_vector_1).mean()

    def cycle_loss_2(self):
        return np.square(self.latent_vector_2 - self.cycle_latent_vector_2).mean()
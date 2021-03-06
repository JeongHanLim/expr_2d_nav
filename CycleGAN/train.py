import torch
import os
import numpy as np
import torch.nn as nn
import pickle as pkl

from tqdm import tqdm
from torch.utils.data import DataLoader
from CycleGAN.model import CycleGAN, Discriminator, CycleGANGen
from CycleGAN.evaluator import Evaluator
from CycleGAN.lr_scheduler import LR_Scheduler
from CycleGAN.dataloader import Transitions
from CycleGAN.summaries import TensorboardSummary

def KLloss(logvar, mu):
    return -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())

class Trainer(object):
    def __init__(self, lr=0.001, epochs=50):
        self.lr = lr
        self.epochs = epochs
        # Define Tensorboard Summary
        self.summary = TensorboardSummary('./')
        self.writer = self.summary.create_summary()
        self.batch_size = 4096
        self.latent_space = 4
        self.gen_iter = 0
        self.dis_iter = 0
        # Define Dataloader
        self.train_set = Transitions('./dataset/1_state.pkl', './dataset/2_state.pkl')
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,)
        self.test_loader = Transitions('./dataset/1_state.pkl', './dataset/2_state.pkl', split='test')

        # Define network
        cyclegan = CycleGAN(latent_space=self.latent_space, state_space=3)
        generator = CycleGANGen(latent_space=self.latent_space, state_space=3)
        discriminator_1 = cyclegan.dis_1
        discriminator_2 = cyclegan.dis_2

        gen_params = [{'params': cyclegan.parameters(), 'lr': self.lr}]
        dis_params_1 = [{'params': discriminator_1.parameters(), 'lr': self.lr}]
        dis_params_2 = [{'params': discriminator_2.parameters(), 'lr': self.lr}]

        # Define Optimizer
        gen_optimizer = torch.optim.Adam(gen_params)
        dis_optimizer_1 = torch.optim.Adam(dis_params_1)
        dis_optimizer_2 = torch.optim.Adam(dis_params_2)

        # Define Criterion
        # whether to use class balanced weights
        self.CycleGANLoss = nn.MSELoss()
        self.DiscLoss = nn.MSELoss()
        self.GANLoss = nn.MSELoss()
        self.VAELoss = nn.MSELoss()

        self.discriminator_1, self.discriminator_2 = discriminator_1, discriminator_2
        self.cyclegan, self.generator, self.gen_optimizer = cyclegan, generator, gen_optimizer
        self.dis_optimizer_1, self.dis_optimizer_2 = dis_optimizer_1, dis_optimizer_2

        # Define Evaluator
        self.evaluator = Evaluator()
        # Define lr scheduler
        self.gen_scheduler = LR_Scheduler(self.lr, self.epochs)
        self.dis_scheduler = LR_Scheduler(self.lr, self.epochs)

        # Using cuda
        # if args.cuda:
        #     self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
        #     self.model = self.model.cuda()

    def gen_training(self, epoch):
        train_loss = 0.0
        self.cyclegan.train()
        self.discriminator_1.requires_grad = False
        self.discriminator_2.requires_grad = False
        tbar = tqdm(self.train_loader)
        for i, sample in enumerate(tbar):
            input_data_1, input_data_2 = sample
            self.gen_scheduler(self.gen_optimizer, epoch)
            self.gen_optimizer.zero_grad()
            labels = torch.ones([input_data_1.shape[0], 1])
            decode_state_1, decode_state_2, cycle_latent_vector_1, cycle_latent_vector_2, \
            latent_vector_1, latent_vector_2, discrim_1, discrim_2, mu_1, mu_2, log_var_1, log_var_2, \
            transfer_latent_vector_1, transfer_latent_vector_2 = self.cyclegan(input_data_1, input_data_2)
            cycle_loss = self.CycleGANLoss(cycle_latent_vector_1, latent_vector_1) + self.CycleGANLoss(cycle_latent_vector_2, latent_vector_2)
            gan_loss = self.GANLoss(discrim_1, labels) + self.GANLoss(discrim_2, labels)
            vae_loss = self.VAELoss(input_data_1, decode_state_1) + self.VAELoss(input_data_2, decode_state_2)
            kl_loss = KLloss(log_var_1, mu_1) + KLloss(log_var_2, mu_2)
            loss = cycle_loss + gan_loss + vae_loss + kl_loss
            loss.backward()
            self.gen_optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.gen_iter += 1
            self.writer.add_scalar('train/total_loss_iter', loss.item(), self.gen_iter)
            self.writer.add_scalar('train/cycle_loss', cycle_loss, self.gen_iter)
            self.writer.add_scalar('train/gan_loss', gan_loss, self.gen_iter)
            self.writer.add_scalar('train/vae_loss', vae_loss, self.gen_iter)

    def dis_training(self, epoch):
        train_loss_1 = 0.0
        train_loss_2 = 0.0
        self.discriminator_1.train()
        self.discriminator_2.train()
        self.discriminator_1.requires_grad = True
        self.discriminator_2.requires_grad = True
        tbar = tqdm(self.train_loader)
        for i, sample in enumerate(tbar):
            self.dis_scheduler(self.dis_optimizer_1, epoch)
            self.dis_scheduler(self.dis_optimizer_2, epoch)
            input_data_1, input_data_2 = sample
            transfer_latent_vector_1, transfer_latent_vector_2, latent_vector_1, latent_vector_2 \
                = self.generator(input_data_1, input_data_2)
            data_1 = torch.cat((transfer_latent_vector_1, latent_vector_1))
            data_2 = torch.cat((transfer_latent_vector_2, latent_vector_2))
            labels = torch.cat((torch.zeros([input_data_1.shape[0], 1]), torch.ones(input_data_1.shape[0], 1)))
            output_1 = self.discriminator_1(data_1)
            output_2 = self.discriminator_2(data_2)
            self.dis_optimizer_1.zero_grad()
            loss_1 = self.DiscLoss(output_1, labels)
            loss_1.backward(retain_graph=True)
            self.dis_optimizer_1.step()

            self.dis_optimizer_2.zero_grad()
            loss_2 = self.DiscLoss(output_2, labels)
            loss_2.backward(retain_graph=True)
            self.dis_optimizer_2.step()

            train_loss_1 += loss_1.item()
            train_loss_2 += loss_2.item()
            self.dis_iter += 1
            self.writer.add_scalar('train/dis_loss_1_iter', loss_1.item(), self.dis_iter)
            self.writer.add_scalar('train/dis_loss_2_iter', loss_2.item(), self.dis_iter)

    # def validation(self, epoch):
    #     self.model.eval()
    #     tbar = tqdm(self.test_loader, desc='\r')
    #     test_loss = 0.0
    #     for i, sample in enumerate(tbar):
    #         image, target = sample['image'], sample['label']
    #         # if self.args.cuda:
    #         #     image, target = image.cuda(), target.cuda()
    #         with torch.no_grad():
    #             output = self.model(image)
    #         loss = self.criterion(output, target)
    #         test_loss += loss.item()
    #         tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
    #         pred = output.data.cpu().numpy()
    #         target = target.cpu().numpy()
    #         pred = np.argmax(pred, axis=1)
    #         # Add batch sample into evaluator
    #         self.evaluator.add_batch(target, pred)
    #
    #     # Fast test during the training
    #     self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)

if __name__ == '__main__':
    args = {'lr'}
    trainer = Trainer(0.001, 150)
    for i in range(50):
        trainer.gen_training(3*i+1)
        trainer.gen_training(3*i+2)
        trainer.gen_training(3*i+3)
        trainer.dis_training(2*i+1)
        trainer.dis_training(2*i+2)
    data1 = torch.from_numpy(trainer.train_set.dataset_1).float()
    data2 = torch.from_numpy(trainer.train_set.dataset_2).float()
    _, _, _, _, latent_vector_1, latent_vector_2, _, _, _, _, _, _, transfer_latent_vector_1, transfer_latent_vector_2= trainer.cyclegan(data1, data2)
    with open('./dataset2/transfer_latent_vector_1.pkl', 'wb') as f:
        pkl.dump(transfer_latent_vector_1.detach().numpy(), f)
    with open('./dataset2/transfer_latent_vector_2.pkl', 'wb') as f:
        pkl.dump(transfer_latent_vector_2.detach().numpy(), f)
    with open('./dataset2/latent_vector_1.pkl', 'wb') as f:
        pkl.dump(latent_vector_1.detach().numpy(), f)
    with open('./dataset2/latent_vector_2.pkl', 'wb') as f:
        pkl.dump(latent_vector_2.detach().numpy(), f)
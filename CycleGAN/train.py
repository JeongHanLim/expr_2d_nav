import torch
import os
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from CycleGAN.model import CycleGAN, Discriminator, CycleGANGen
from CycleGAN.evaluator import Evaluator
from CycleGAN.lr_scheduler import LR_Scheduler
from CycleGAN.dataloader import Transitions
from CycleGAN.summaries import TensorboardSummary

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Tensorboard Summary
        self.summary = TensorboardSummary('./')
        self.writer = self.summary.create_summary()

        # Define Dataloader
        self.train_loader = Transitions('./')
        self.test_loader = Transitions('./', split='test')

        # Define network
        cyclegan = CycleGAN(latent_space=16, state_space=3)
        generator = CycleGANGen(latent_space=16, state_space=3)
        discriminator_1 = Discriminator(latent_space=16)
        discriminator_2 = Discriminator(latent_space=16)

        gen_params = [{'params': cyclegan.parameters(), 'lr': args.lr}]
        dis_params_1 = [{'params': discriminator_1.parameters(), 'lr': args.lr}]
        dis_params_2 = [{'params': discriminator_2.parameters(), 'lr': args.lr}]

        # Define Optimizer
        gen_optimizer = torch.optim.Adam(gen_params)
        dis_optimizer_1 = torch.optim.Adam(dis_params_1)
        dis_optimizer_2 = torch.optim.Adam(dis_params_2)

        # Define Criterion
        # whether to use class balanced weights
        self.CycleGANLoss = nn.MSELoss()
        self.DiscLoss = nn.MSELoss()
        self.GANLoss = nn.BCELoss()
        self.VAELoss = nn.MSELoss()

        self.discriminator_1, self.discriminator_2 = discriminator_1, discriminator_2
        self.cyclegan, self.generator, self.gen_optimizer = cyclegan, generator, gen_optimizer
        self.dis_optimizer_1, self.dis_optimizer_2 = dis_optimizer_1, dis_optimizer_2

        # Define Evaluator
        self.evaluator = Evaluator()
        # Define lr scheduler
        self.gen_scheduler = LR_Scheduler(args.lr, args.epochs)
        self.dis_scheduler = LR_Scheduler(args.lr, args.epochs)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()

    def gen_training(self, epoch):
        train_loss = 0.0
        self.cyclegan.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            input_data_1, input_data_2 = sample
            self.gen_scheduler(self.gen_optimizer, epoch)
            self.gen_optimizer.zero_grad()
            decode_state_1, decode_state_2, cycle_latent_vector_1, cycle_latent_vector_2, \
            latent_vector_1, latent_vector_2, discrim_1, discrim_2 = self.model(input_data_1, input_data_2)
            cycle_loss = self.CycleGANLoss(cycle_latent_vector_1, latent_vector_1) + self.CycleGANLoss(cycle_latent_vector_2, latent_vector_2)
            gan_loss = self.GANLoss(discrim_1, discrim_2)
            vae_loss = self.VAELoss(input_data_1, decode_state_1) + self.VAELoss(input_data_2, decode_state_2)
            loss = cycle_loss + gan_loss + vae_loss
            loss.backward()
            self.gen_optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)

    def dis_training(self, epoch):
        train_loss_1 = 0.0
        train_loss_2 = 0.0

        self.discriminator_1.train()
        self.discriminator_2.train()

        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            self.dis_scheduler(self.dis_optimizer_1, epoch)
            self.dis_scheduler(self.dis_optimizer_2, epoch)
            input_data_1, input_data_2 = sample
            transfer_latent_vector_1, transfer_latent_vector_2, latent_vector_1, latent_vector_2 \
                = self.generator(input_data_1, input_data_2)
            data_1 = torch.cat(transfer_latent_vector_1, latent_vector_1)
            data_2 = torch.cat(transfer_latent_vector_2, latent_vector_2)
            labels = torch.cat((torch.zeros(), torch.ones()))
            output_1 = self.discriminator_1(data_1)
            output_2 = self.discriminator_2(data_2)
            self.dis_optimizer_1.zero_grad()
            loss_1 = self.DiscLoss(output_1, labels)
            loss_1.backward()
            self.dis_optimizer_1.step()

            self.dis_optimizer_2.zero_grad()
            loss_2 = self.DiscLoss(output_2, labels)
            loss_2.backward()
            self.dis_optimizer_2.step()

            train_loss_1 += loss_1.item()
            train_loss_2 += loss_2.item()
            self.writer.add_scalar('train/dis_loss_1_iter', loss_1.item(), i + num_img_tr * epoch)
            self.writer.add_scalar('train/dis_loss_2_iter', loss_2.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss_1, epoch)
        self.writer.add_scalar('train/total_loss_epoch', train_loss_2, epoch)

    def validation(self, epoch):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)

if __name__ == '__main__':
    args = {}
    trainer = Trainer(args)
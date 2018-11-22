import os
import numpy as np
import argparse
import torch
import time
import pickle
from trainingDataset import trainingDataset
from model import Generator, Discriminator


class CycleGANTraining:
    def __init__(self, logf0s_normalization, mcep_normalization, coded_sps_A_norm, coded_sps_B_norm):
        self.num_epochs = 5000
        self.mini_batch_size = 1
        self.dataset_A = self.loadPickleFile(coded_sps_A_norm)
        self.dataset_B = self.loadPickleFile(coded_sps_B_norm)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Generator and Discriminator
        self.generator_A2B = Generator().to(self.device)
        self.generator_B2A = Generator().to(self.device)
        self.discriminator_A = Discriminator().to(self.device)
        self.discriminator_B = Discriminator().to(self.device)

        # Loss Functions
        criterion_mse = torch.nn.MSELoss()

        # Optimizer
        g_params = list(self.generator_A2B.parameters()) + \
            list(self.generator_B2A.parameters())
        d_params = list(self.discriminator_A.parameters()) + \
            list(self.discriminator_B.parameters())

        self.generator_optimizer = torch.optim.Adam(
            g_params, lr=0.0002, betas=(0.5, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(
            d_params, lr=0.0001, betas=(0.5, 0.999))

    def reset_grad(self):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

    def train(self):
        # Training Begins
        for epoch in range(self.num_epochs):
            start_time_epoch = time.time()

            # Constants
            cycle_loss_lambda = 10
            identity_loss_lambda = 5

            # Preparing Dataset
            dataset = trainingDataset(datasetA=self.dataset_A,
                                      datasetB=self.dataset_B,
                                      n_frames=128)
            train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=self.mini_batch_size,
                                                       shuffle=True,
                                                       drop_last=False)

            for i, (real_A, real_B) in enumerate(train_loader):

                real_A = real_A.to(self.device).float()
                real_B = real_B.to(self.device).float()

                # Generator Loss function
                self.reset_grad()
                fake_B = self.generator_A2B(real_A)
                cycle_A = self.generator_B2A(fake_B)

                fake_A = self.generator_B2A(real_B)
                cycle_B = self.generator_A2B(fake_A)

                identity_A = self.generator_B2A(real_A)
                identity_B = self.generator_A2B(real_B)

                d_fake_A = self.discriminator_A(fake_A)
                d_fake_B = self.discriminator_B(fake_B)

                # Generator Cycle loss
                cycleLoss = torch.mean(
                    torch.abs(real_A - cycle_A)) + torch.mean(torch.abs(real_B - cycle_B))

                # Generator Identity Loss
                identiyLoss = torch.mean(
                    torch.abs(real_A - identity_A)) + torch.mean(torch.abs(real_B - identity_B))

                # Generator Loss
                generator_loss_A2B = torch.mean((1 - d_fake_B)**2)
                generator_loss_B2A = torch.mean((1 - d_fake_A)**2)

                # Total Generator Loss
                generator_loss = generator_loss_A2B + generator_loss_B2A + \
                    cycle_loss_lambda * cycleLoss + identity_loss_lambda * identiyLoss

                # Backprop for Generator
                generator_loss.backward()
                self.generator_optimizer.step()

                # Discriminator Loss Function
                self.reset_grad()

                # Discriminator Feed Forward
                d_real_A = self.discriminator_A(real_A)
                d_real_B = self.discriminator_B(real_B)

                generated_A = self.generator_B2A(real_B)
                d_fake_A = self.discriminator_A(generated_A)

                generated_B = self.generator_A2B(real_A)
                d_fake_B = self.discriminator_B(generated_B)

                # Loss Functions
                d_loss_A_real = torch.mean((1 - d_real_A)**2)
                d_loss_A_fake = torch.mean((0 - d_fake_A)**2)
                d_loss_A = (d_loss_A_real + d_loss_A_fake) / 2.0
                # print("D loss -> ", d_loss_A)
                d_loss_B_real = torch.mean((1 - d_real_B)**2)
                d_loss_B_fake = torch.mean((0 - d_fake_B)**2)
                d_loss_B = (d_loss_B_real + d_loss_B_fake) / 2.0

                # Final Loss for discriminator
                d_loss = (d_loss_A + d_loss_B) / 2.0

                # Backprop for Discriminator
                d_loss.backward()
                self.discriminator_optimizer.step()
            end_time = time.time()
            print("Iter: {} Generator Loss: {:.4f} Discriminator Loss: {}".format(
                epoch, generator_loss.item(), d_loss.item()))

    def savePickle(self, variable, fileName):
        with open(fileName, 'wb') as f:
            pickle.dump(variable, f)

    def loadPickleFile(self, fileName):
        with open(fileName, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train CycleGAN using source dataset and target dataset")

    logf0s_normalization_default = '../cache/logf0s_normalization.npz'
    mcep_normalization_default = '../cache/mcep_normalization.npz'
    coded_sps_A_norm = '../cache/coded_sps_A_norm.pickle'
    coded_sps_B_norm = '../cache/coded_sps_B_norm.pickle'

    parser.add_argument('--logf0s_normalization', type=str,
                        help="Cached location for log f0s normalized", default=logf0s_normalization_default)
    parser.add_argument('--mcep_normalization', type=str,
                        help="Cached location for mcep normalization", default=mcep_normalization_default)
    parser.add_argument('--coded_sps_A_norm', type=str,
                        help="mcep norm for data A", default=coded_sps_A_norm)
    parser.add_argument('--coded_sps_B_norm', type=str,
                        help="mcep norm for data B", default=coded_sps_B_norm)

    argv = parser.parse_args()

    logf0s_normalization = argv.logf0s_normalization
    mcep_normalization = argv.mcep_normalization
    coded_sps_A_norm = argv.coded_sps_A_norm
    coded_sps_B_norm = argv.coded_sps_B_norm

    # Check whether following cached files exists
    if not os.path.exists(logf0s_normalization) or not os.path.exists(mcep_normalization):
        print(
            "Cached files do not exist, please run the program preprocess_training.py first")

    cycleGAN = CycleGANTraining(logf0s_normalization=logf0s_normalization,
                                mcep_normalization=mcep_normalization,
                                coded_sps_A_norm=coded_sps_A_norm,
                                coded_sps_B_norm=coded_sps_B_norm)
    cycleGAN.train()

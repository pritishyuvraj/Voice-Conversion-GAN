import torch
from torch import optim
from torch.autograd import Variable

# Custom Libraries
from datasets import MNISTdataset, SVHNdataset
import models


class solver:
    def __init__(self, learning_rate=0.0002):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.MNISTdataset = MNISTdataset(batch_size=128, image_size=32)
        self.SVHNdataset = SVHNdataset(batch_size=128, image_size=32)
        self.G12 = models.G12().to(self.device)
        self.G21 = models.G21().to(self.device)
        self.D1 = models.D1().to(self.device)
        self.D2 = models.D2().to(self.device)

        self.g_params = list(self.G12.parameters()) + \
            list(self.G21.parameters())
        self.d_params = list(self.D1.parameters()) + list(self.D2.parameters())

        self.g_optimizer = optim.Adam(self.g_params, learning_rate)
        self.d_optimizer = optim.Adam(self.d_params, learning_rate)

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def train(self):
        for i in range(100):
            for index, ((MNISTimage, _), (SVHNimage, _)) in enumerate(
                    zip(self.MNISTdataset.MNISTdataLoader, self.SVHNdataset.SVHNdataLoader)):
                MNISTimage = MNISTimage.to(self.device)
                SVHNimage = SVHNimage.to(self.device)
                # print("see this -> ",i, index,  MNISTimage.shape, SVHNimage.shape)

                self.reset_grad()
                out = self.D1(MNISTimage)
                d1_loss = torch.mean((out - 1)**2)

                out = self.D2(SVHNimage)
                d2_loss = torch.mean((out - 1)**2)

                d_real_loss = d1_loss + d2_loss
                d_real_loss.backward()
                self.d_optimizer.step()

                # Train with Fake Images
                self.reset_grad()
                fake_MNIST = self.G21(SVHNimage)
                out = self.D1(fake_MNIST)
                d1_loss = torch.mean(out**2)

                fake_SVHN = self.G12(MNISTimage)
                out = self.D2(fake_SVHN)
                d2_loss = torch.mean(out**2)

                d_fake_loss = d1_loss + d2_loss
                d_fake_loss.backward()
                self.d_optimizer.step()

                # Train the Generator

                # Train MNIST - SVHN - MNIST Cycle:
                self.reset_grad()
                fake_SVHN = self.G12(MNISTimage)
                out = self.D2(fake_SVHN)
                reconstruct_MNIST = self.G21(fake_SVHN)
                g_loss_C1 = torch.mean((out - 1)**2)
                g_loss_C1 += torch.mean((reconstruct_MNIST - MNISTimage)**2)
                g_loss_C1.backward()
                self.g_optimizer.step()

                # Train SVHN - MNIST - SVHN Cycle:
                self.reset_grad()
                fake_MNIST = self.G21(SVHNimage)
                out = self.D1(fake_MNIST)
                reconstruct_SVHN = self.G12(fake_MNIST)
                g_loss_C2 = torch.mean((out - 1)**2)
                g_loss_C2 += torch.mean((reconstruct_SVHN - SVHNimage)**2)
                g_loss_C2.backward()
                self.g_optimizer.step()

                print("Loss Real Images: {:.4f}, Loss Fake Images: {:.4f}, Loss MNIST-SVHN-MNIST: {:.4f}, Loss SVHN-MNIST-SVHN: {:.4f}".format(
                    d_real_loss.item(), d_fake_loss.item(), g_loss_C1.item(), g_loss_C2.item()))

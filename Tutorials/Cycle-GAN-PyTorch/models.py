import torch
import torch.nn as nn


class G12(nn.Module):
    # Convert MNIST to SVHN
    def __init__(self, depth=64):
        super(G12, self).__init__()
        # Encoding Layer
        self.conv1 = nn.Sequential(nn.Conv2d(1, depth, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(depth),
                                   nn.LeakyReLU(0.05))
        self.conv2 = nn.Sequential(nn.Conv2d(depth, depth * 2, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(depth * 2),
                                   nn.LeakyReLU(0.05))

        # Residual Blocks
        self.conv3 = nn.Sequential(nn.Conv2d(depth * 2, depth * 2, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(depth * 2),
                                   nn.LeakyReLU(0.05))
        self.conv4 = nn.Sequential(nn.Conv2d(depth * 2, depth * 2, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(depth * 2),
                                   nn.LeakyReLU(0.05))

        # Decoding Block
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(depth * 2, depth, kernel_size=4, stride=2, padding=1),
                                     nn.BatchNorm2d(depth),
                                     nn.LeakyReLU(0.05))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(depth, 3, kernel_size=4, stride=2, padding=1),
                                     nn.Tanh())

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.deconv1(out)
        out = self.deconv2(out)
        return out


class G21(nn.Module):
    # Convert SVHM to MNIST
    def __init__(self, depth=64):
        super(G21, self).__init__()
        # Encoding Layer
        self.conv1 = nn.Sequential(nn.Conv2d(3, depth, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(depth),
                                   nn.LeakyReLU(0.05))
        self.conv2 = nn.Sequential(nn.Conv2d(depth, depth * 2, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(depth * 2),
                                   nn.LeakyReLU(0.05))

        # Residual Block
        self.conv3 = nn.Sequential(nn.Conv2d(depth * 2, depth * 2, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(depth * 2),
                                   nn.LeakyReLU(0.05))
        self.conv4 = nn.Sequential(nn.Conv2d(depth * 2, depth * 2, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(depth * 2),
                                   nn.LeakyReLU(0.05))

        # Decoding Block
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(depth * 2, depth, kernel_size=4, stride=2, padding=1),
                                     nn.BatchNorm2d(depth),
                                     nn.LeakyReLU(0.05))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(depth, 1, kernel_size=4, stride=2, padding=1),
                                     nn.Tanh())

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.deconv1(out)
        out = self.deconv2(out)
        return out

class D1(nn.Module):
    # Discriminator for MNIST Image
    def __init__(self, depth = 64):
        super(D1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, depth, kernel_size=4, padding=1, stride=2),
                                   nn.LeakyReLU(0.05))
        self.conv2 = nn.Sequential(nn.Conv2d(depth, depth*2, kernel_size=4, padding=1, stride=2),
                                   nn.BatchNorm2d(depth*2),
                                   nn.LeakyReLU(0.05))
        self.conv3 = nn.Sequential(nn.Conv2d(depth*2, depth*4, kernel_size=4, padding=1, stride=2),
                                   nn.BatchNorm2d(depth*4),
                                   nn.LeakyReLU(0.05))
        self.fc = nn.Sequential(nn.Conv2d(depth*4, 1, kernel_size=4, padding=0, stride=1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc(out).squeeze()
        return out

class D2(nn.Module):
    # Discriminator for SVHM Image
    def __init__(self, depth = 64):
        super(D2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, depth, kernel_size=4, padding=1, stride=2),
                                   nn.LeakyReLU(0.05))
        self.conv2 = nn.Sequential(nn.Conv2d(depth, depth*2, kernel_size=4, padding=1, stride=2),
                                   nn.BatchNorm2d(depth*2),
                                   nn.LeakyReLU(0.05))
        self.conv3 = nn.Sequential(nn.Conv2d(depth*2, depth*4, kernel_size=4, padding=1, stride=2),
                                   nn.BatchNorm2d(depth*4),
                                   nn.LeakyReLU(0.05))
        self.fc = nn.Sequential(nn.Conv2d(depth*4, 1, kernel_size=4, padding=0, stride=1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc(out).squeeze()
        return out

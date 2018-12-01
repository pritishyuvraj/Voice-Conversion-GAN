import torch.nn as nn
import torch
import numpy as np


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
        # Custom Implementation because the Voice Conversion Cycle GAN
        # paper assumes GLU won't reduce the dimension of tensor by 2.

    def forward(self, input):
        return input * torch.sigmoid(input)


class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        # Custom Implementation because PyTorch PixelShuffle requires,
        # 4D input. Whereas, in this case we have have 3D array
        self.upscale_factor = upscale_factor

    def forward(self, input):
        n = input.shape[0]
        c_out = input.shape[1] // 2
        w_new = input.shape[2] * 2
        return input.view(n, c_out, w_new)


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualLayer, self).__init__()

        self.residualLayer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                     out_channels=out_channels,
                                                     kernel_size=kernel_size,
                                                     stride=1,
                                                     padding=padding),
                                           nn.InstanceNorm1d(
                                               num_features=out_channels),
                                           GLU(),
                                           nn.Conv1d(in_channels=out_channels,
                                                     out_channels=in_channels,
                                                     kernel_size=kernel_size,
                                                     stride=1,
                                                     padding=padding),
                                           nn.InstanceNorm1d(
                                               num_features=in_channels)
                                           )

    def forward(self, input):
        return input + self.residualLayer(input)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=24,
                                             out_channels=128,
                                             kernel_size=15,
                                             stride=1,
                                             padding=7),
                                   GLU())

        # Downsample Layer
        self.downSample1 = self.downSample(in_channels=128,
                                           out_channels=256,
                                           kernel_size=5,
                                           stride=2,
                                           padding=1)

        self.downSample2 = self.downSample(in_channels=256,
                                           out_channels=512,
                                           kernel_size=5,
                                           stride=2,
                                           padding=2)

        # Residual Blocks
        self.residualLayer = ResidualLayer(in_channels=512,
                                           out_channels=1024,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)

        # UpSample Layer
        self.upSample1 = self.upSample(in_channels=512,
                                       out_channels=1024,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)

        self.upSample2 = self.upSample(in_channels=1024 // 2,
                                       out_channels=512,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)

        self.lastConvLayer = nn.Conv1d(in_channels=512 // 2,
                                       out_channels=24,
                                       kernel_size=15,
                                       stride=1,
                                       padding=7)

    def downSample(self, in_channels, out_channels,  kernel_size, stride, padding):
        self.ConvLayer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm1d(
                                       num_features=out_channels),
                                       GLU())

        return self.ConvLayer

    def upSample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.convLayer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       PixelShuffle(upscale_factor=2),
                                       nn.InstanceNorm1d(
                                           num_features=out_channels // 2),
                                       GLU())
        return self.convLayer

    def forward(self, input):
        conv1 = self.conv1(input)
        downsample1 = self.downSample1(conv1)
        downsample2 = self.downSample2(downsample1)
        residual_layer_1 = self.residualLayer(downsample2)
        residual_layer_2 = self.residualLayer(residual_layer_1)
        residual_layer_3 = self.residualLayer(residual_layer_2)
        residual_layer_4 = self.residualLayer(residual_layer_3)
        residual_layer_5 = self.residualLayer(residual_layer_4)
        residual_layer_6 = self.residualLayer(residual_layer_5)
        residual_layer_7 = self.residualLayer(residual_layer_6)
        upSample_layer_1 = self.upSample1(residual_layer_7)
        upSample_layer_2 = self.upSample2(upSample_layer_1)
        output = self.lastConvLayer(upSample_layer_2)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.convLayer1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                                  out_channels=128,
                                                  kernel_size=[3, 4],
                                                  stride=[1, 2],
                                                  padding=[1, 1]),
                                        GLU())

        # Note: Kernel Size have been modified in the PyTorch implementation
        # compared to the actual paper, as to retain dimensionality. Unlike,
        # TensorFlow, PyTorch doesn't have padding='same', hence, kernel sizes
        # were altered to retain the dimensionality after each layer

        # DownSample Layer
        self.downSample1 = self.downSample(in_channels=128,
                                           out_channels=256,
                                           kernel_size=[4, 4],
                                           stride=[2, 2],
                                           padding=1)

        self.downSample2 = self.downSample(in_channels=256,
                                           out_channels=512,
                                           kernel_size=[4, 4],
                                           stride=[2, 2],
                                           padding=1)

        self.downSample3 = self.downSample(in_channels=512,
                                           out_channels=1024,
                                           kernel_size=[5, 4],
                                           stride=[1, 2],
                                           padding=[2, 1])

        # Fully Connected Layer
        self.fc = nn.Linear(in_features=1024,
                            out_features=1)

    def downSample(self, in_channels, out_channels, kernel_size, stride, padding):
        convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding),
                                  nn.InstanceNorm2d(num_features=out_channels),
                                  GLU())
        return convLayer

    def forward(self, input):
        # input has shape [batch_size, num_features, time]
        # discriminator requires shape [batchSize, 1, num_features, time]
        input = input.unsqueeze(1)
        layer1 = self.convLayer1(input)
        downSample1 = self.downSample1(layer1)
        downSample2 = self.downSample2(downSample1)
        downSample3 = self.downSample3(downSample2)
        downSample3 = downSample3.contiguous().permute(0, 2, 3, 1).contiguous()
        fc = torch.sigmoid(self.fc(downSample3))
        return fc


if __name__ == '__main__':
    # Generator Dimensionality Testing
    input = torch.randn(10, 24, 1100)  # (N, C_in, Width) For Conv1d
    np.random.seed(0)
    print(np.random.randn(10))
    input = np.random.randn(158, 24, 128)
    input = torch.from_numpy(input).float()
    print(input)
    generator = Generator()
    output = generator(input)
    print("Output shape Generator", output)

    # Discriminator Dimensionality Testing
    # input = torch.randn(32, 1, 24, 128)  # (N, C_in, height, width) For Conv2d
    discriminator = Discriminator()
    output = discriminator(output)
    print("Output shape Discriminator", output.shape)

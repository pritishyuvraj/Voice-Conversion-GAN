import torch.nn as nn
import torch


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
        # Custom Implementation because the Voice Conversion Cycle GAN
        # paper assumes GLU won't reduce the dimension of tensor by 2. (
        # Which is weird!!)

    def forward(self, input):
        return input * torch.sigmoid(input)


class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        # Custom Implementation because PyTorch PixelShuffle requires,
        # 4D input. Whereas, in this case we have have 3D arry
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


if __name__ == '__main__':
    input = torch.randn(1, 24, 128)
    generator = Generator()
    output = generator(input)
    print("Output shape", output.shape)

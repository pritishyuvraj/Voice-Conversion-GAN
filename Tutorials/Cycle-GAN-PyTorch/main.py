import torch.nn

# Custom Libraries
from solver import solver

class cycleLossGAN:
    def __init__(self):
        # Dataset Loader
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.solver = solver(learning_rate = 0.0002)

        # self.MNISTdataset = MNISTdataset(batch_size=128, image_size=32)
        # self.SVHNdataset = SVHNdataset(batch_size=128, image_size=32)
        # self.G12 = models.G12().to(self.device)
        # self.G21 = models.G21().to(self.device)
        # self.D1 = models.D1().to(self.device)
        # self.D2 = models.D2().to(self.device)

    def fit(self):
        # for index, ((MNISTimage, _), (SVHNimage, _)) in enumerate(
        #         zip(self.MNISTdataset.MNISTdataLoader, self.SVHNdataset.SVHNdataLoader)):
        #     MNISTimage = MNISTimage.to(self.device)
        #     SVHNimage = SVHNimage.to(self.device)
        #
        #     print(index, MNISTimage.shape, SVHNimage.shape)
        #     SVHNoutput = self.G12(MNISTimage)
        #     MNISToutput = self.G21(SVHNimage)
        #     outputMNIST = self.D1(MNISToutput)
        #     outputSVHN = self.D2(SVHNoutput)
        #     print(SVHNoutput.shape, MNISToutput.shape, outputMNIST.shape,outputSVHN.shape )
        self.solver.train()

if __name__ == '__main__':
    clGAN = cycleLossGAN()
    clGAN.fit()

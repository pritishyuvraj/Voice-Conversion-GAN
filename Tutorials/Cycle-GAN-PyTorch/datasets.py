import torchvision
from torchvision import transforms
import torch

class MNISTdataset:
    def __init__(self, batch_size = 128, image_size = 32):
        transform = transforms.Compose([transforms.Scale(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                             std=(0.5, 0.5, 0.5))])

        self.dataset = torchvision.datasets.MNIST(root = '../../../data/',
                                                  transform = transform,
                                                  download = True)

        self.MNISTdataLoader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                      batch_size = batch_size,
                                                      shuffle=True,
                                                      drop_last=True)

    def testMNISTdataset(self):
        for i, (images, _) in enumerate(self.MNISTdataLoader):
            print(images.shape, _.shape)
            if i == 2: break

class SVHNdataset:
    def __init__(self, batch_size = 128, image_size = 32):
        transform = transforms.Compose([transforms.Scale(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                             std=(0.5, 0.5, 0.5))])

        dataset = torchvision.datasets.SVHN(root="../../../data/",
                                            transform=transform,
                                            download=True)

        self.SVHNdataLoader = torch.utils.data.DataLoader(dataset=dataset,
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          drop_last=True)

    def testSVHNdataset(self):
        for i, (images, _) in enumerate(self.SVHNdataLoader):
            print(images.shape, _.shape)
            if i == 2: break

if __name__ == '__main__':
    mnist = MNISTdataset()
    mnist.testMNISTdataset()

    svhn = SVHNdataset()
    svhn.testSVHNdataset()

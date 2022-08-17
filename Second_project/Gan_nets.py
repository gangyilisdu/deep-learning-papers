import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsts
import torchvision.utils as vutils
import torchvision.transforms as transforms


def sample_from_dataset(batch_size, image_shape, data_dir, data):
    sample_dim = (batch_size,) + image_shape
    sample = np.empty(sample_dim, dtype = float.32)
    all_data_dirlist = list(glob.glob(data_dir))
    sample_imgs_paths = np.random.choice(all_data_dirlist, batch_size)
    for index, img_filename in enumerate(sample_imgs_paths):




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.ConvT1 = nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False)
        self.BN1 = nn.BatchNorm2d(512)
        self.Relu1 = nn.ReLU()

        self.ConvT2 = nn.ConvTranspose2d(512, 256, 4, 1, 0, bias=False)
        self.BN2 = nn.BatchNorm2d(256)
        self.Relu2 = nn.ReLU()

        self.ConvT3 = nn.ConvTranspose2d(256, 128, 4, 1, 0, bias = False)
        self.BN3 = nn.BatchNorm2d(128)
        self.Relu3 = nn.ReLU()

        self.ConvT4 = nn.ConvTranspose2d(128, 64, 4, 1, 0, bias = False)
        self.BN4 = nn.BatchNorm2d(64)
        self.Relu4 = nn.ReLU()

        self.ConvT5 = nn.ConvTranspose2d(64, 3, 4, 1, 0, bias=False)
        self.Tanh = nn.Tanh()

    def forward(self, input):
        x = input
        x = self.ConvT1(x)
        x = self.BN1(x)
        x = self.Relu1(x)

        x = self.ConvT2(x)
        x = self.BN2(x)
        x = self.Relu2(x)

        x = self.ConvT3(x)
        x = self.BN3(x)
        x = self.Relu3(x)

        x = self.ConvT4(x)
        x = self.BN4(x)
        x = self.Relu4(x)

        x = self.ConvT5(x)
        x = self.Tanh(x)

        return x

netG = Generator()
netG.apply(weights_init)
print(netG)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(

            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.main(x)
        return output


netD = Discriminator()

noise = torch.randn(1, 100, 1, 1)


print(netG(noise))
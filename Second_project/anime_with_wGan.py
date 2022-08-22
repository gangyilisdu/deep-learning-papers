import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torch.utils.data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


image_size = 64
transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


datapath = 'E:\\workstuff\\animie\\animeface-character-dataset\\animeface-character-dataset'
Img_datasets = datasets.ImageFolder(os.path.join(datapath), transforms)
dataloader = torch.utils.data.DataLoader(Img_datasets, 64, shuffle = True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.ConvT1 = nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False)
        self.BN1 = nn.BatchNorm2d(512)
        self.Relu1 = nn.ReLU()

        self.ConvT2 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.BN2 = nn.BatchNorm2d(256)
        self.Relu2 = nn.ReLU()

        self.ConvT3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False)
        self.BN3 = nn.BatchNorm2d(128)
        self.Relu3 = nn.ReLU()

        self.ConvT4 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False)
        self.BN4 = nn.BatchNorm2d(64)
        self.Relu4 = nn.ReLU()

        self.ConvT5 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
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
        x = self.Tanh(x)

        x = self.ConvT5(x)
        x = self.Tanh(x)

        return x

netG = Generator().to(device)
netG.apply(weights_init)
#print(netG)

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

            nn.Conv2d(512, 1, 4, 1, 0, bias = False),
            #nn.Sigmoid()
        )
    def forward(self, input):
        output = self.main(input)
        return output


netD = Discriminator().to(device)

netD.apply(weights_init)

optimizerG = optim.RMSprop(netG.parameters(), lr = 0.0001)
optimizerD = optim.RMSprop(netD.parameters(), lr = 0.0001)
fixed_noise = torch.randn(64, 100, 1, 1, device = device)

num_epochs = 100

img_list = []
G_losses, D_losses = [], []
iters = 0


print('Staring Training Loop')

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)

        output = netD(real_cpu).view(-1)
        errD_real = -torch.mean(output)

        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size, 100, 1, 1, device = device)
        fake = netG(noise)

        output = netD(fake.detach()).view(-1)
        errD_fake = torch.mean(output)

        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_fake + errD_real

        optimizerD.step()

        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)

        netG.zero_grad()

        output = netD(fake).view(-1)
        errG = -torch.mean(output)
        errG.backward()
        D_G_z2 = output.mean().item()

        optimizerG.step()

        if i % 50 == 0:
            print('[%d / %d][%d / %d]\tLoss_D: %.4f\tLoss_G : %.4f\tD(x) : %.4f\tD(G(z)) : %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if iters % 500  == 0 or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding = 2, normalize = True))

        iters += 1


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()











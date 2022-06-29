### Generator
import torch
import os
import random 
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim 
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

# Root directory for dataset
# the path to the root of the dataset folder. 
dataroot = "data/celeba"

# Number of workers for dataloader
# the number of worker threads for loading the Data with the DataLoader 
workers = 2

# Batch size during training
# the batch size used in training, The DCGAN paper uses a batch size of 128 
batch_size = 128

# Spatial size of training image. All images will be resized to this 
# size using a transformer.
# This implementatio defaults to 64 * 64. If another size is desired, the structures of D and G 
# must be changed 
image_size = 64

# Numebr of channels in the training images. For the color image this is 3
nc = 3

# Size of z latent vector (i.e, size of generator input)
nz = 100

# Size of feature maps in generator
# related to the depth of feature maps propagated through the generator
ngf = 64

# Size of feature maps in discriminator
# sets the depth of feature maps propagated through the discriminator 
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers 
# As described in the DCGAN paper, this number should be 0.0002
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
# As described in the DCGAN paper, this number should be 0.5
beta1 = 0.5

# Number of GPUs available, Use 0 for CPU mode.
ngpu = 0 



### Celeb-A Faces dataset
dataset = dset.ImageFolder(root = dataroot, 
                           transform = transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))   

# Create the dataloader
dataloader = torch.utils.data.Dataloader(dataset, batch_size = batch_size, shuffle = True, num_workers = workers)

# Device which device we want to run on
device = torch.deivce("cuda:0" if (torch.cuda.is_avaiable() and ngpu > 0) else 'cpu')




# Weight Initialization
# From the DCGAN paper, the authors specify that all model weights shall be randomly initialized
# from a Normal distribution with mean = 0, stdev = 0.02. The weight_init functions takes an
# initalized model as input and reinitializes all convolution, covolution-transpose, and batch normalization
# layer to meet this criteria. This function is applied to the model immediately after initialization.

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 : 
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 : 
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf* 8, 4, 1, 0, bias = False),
            nn.BatchNorm2d(ngf* 8),
            nn.ReLU(True),   
            
            # state size. (ngf * 8) * 4 * 4
            nn.ConvTranspose2d(ngf* 8, ngf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True), 

            # state size. (ngf * 4) * 8  * 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True), 

            # state size. (ngf * 2) * 16 * 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias = False),
            nn.Tanh()
            
            # state size. (nc) * 64 * 64 
        )
    def forward(self, input):
        output = self.main(input)
        return output

netG = Generator()
# Apply the weights_init function to randomly initalize all weights.
# to mean = 0, stdev = 0.02 
netG.apply(weights_init)
print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        self.main = nn.Sequential(
            # input is (nc) * 64 *64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            # start size, (ndf) * 32 * 32
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),
            # state size, (ndf) * 16 * 16
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            # state size, (ndf)* 8 * 8
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace = True),
            # state size, (ndf) * 4 *4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)
    

# create the discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired 
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to random initialize all weights
# to mean = 0, stdev = 0.2
netD.apply(weights_init)

print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
# the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device = device)

# Establish convention for real and fake labels during training
real_label = 1.0
fake_label = 0.0

# Setup Adam optimizers for both G and D
optimizerG = optim.Adam(netG.parameters(), lr = lr, betas = (beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr = lr, betas = (beta1, 0.999))


# Training Loop

# Lists to keep track of progress

img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype = torch.float, device = device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device = device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on tha all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradient for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label) # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1























    
    
    
    
    
    
    
    
    
    
    

        


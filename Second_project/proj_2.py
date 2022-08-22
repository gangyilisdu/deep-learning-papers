import csv
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_file_path = "dataset_4/dataset_4/input_4.csv"
output_file_path = "dataset_4/dataset_4/result_4.csv"
adj = list(csv.reader(open('E:\workstuff\Code table.csv')))
conn = list(csv.reader(open('E:\workstuff\sequence.csv')))


'''
conn = []
for j, i in enumerate(adj):
    conn.append([j])
    for index, item in enumerate(i):
        if eval(item) == 1 and index != j:
            conn[-1].append(index)
conn = conn[1:]
del conn[0][1]
print(conn)
'''
print(conn)
class DiyDataset(Dataset):
    def __init__(self, input_file_path, out_file_path):
        x = np.loadtxt(input_file_path, delimiter= ',', dtype=np.float32)
        y = np.loadtxt(out_file_path, delimiter= ',', dtype=np.float32)
        self.len = 45
        mu = np.mean(y)
        sigma = np.std(y)
        ydata = (y - mu)/sigma
        self.x_data = torch.from_numpy(x.T)
        self.y_data = torch.from_numpy(ydata.T)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len

input_file_path = "dataset_4/dataset_4/input_4.csv"
output_file_path = "dataset_4/dataset_4/result_4.csv"
D_dataset = DiyDataset(input_file_path, output_file_path)
D_dataLoader = DataLoader(dataset=D_dataset, batch_size = D_dataset.len, shuffle=True)
print(len(D_dataLoader))
'''
kn = torch.rand(1, 1, 4)
print(kn.size())
'''

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv2') != -1 :
        nn.init.constant_(m.weight.data, 1.0)

def collect_and_transfer(input):
    output = []
    for i in range(168):
        adj = conn[i]
        batch_size = input.size(0)
        input_value = []
        for j in adj:
            current_batch_input_value = []
            for batch in range(batch_size):
                current_batch_input_value.append(input[batch][int(j) - 1].item())
            input_value.append(current_batch_input_value)
        if len(input_value) == 3:
            input_value.append([0.0] * 45)
        input_value = np.array(input_value, dtype=np.float32).T
        # input_value = torch.from_numpy(input_value)
        # input_value = input_value.view(45, 1 ,1, 4)
        # res = self.conv1(input_value)
        output.append(input_value)
    output = torch.tensor([item for item in output])
    output = output.view(45, 1, 4, 168)
    return output


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size= (1, 4), stride= (1, 4))
        self.conv1.weight.data = Variable(torch.Tensor([[[[0.2,  0.1, 0.1,  0.1]], [[0.2, 0.1, 0.1,  0.1]]]]))
        self.conv1.weight.requires_grad = True
        self.relu = nn.ReLU()
        self.Linear1 = nn.Linear(168, 168)
    def forward(self, input, label):
        output_input = collect_and_transfer(input)
        label_input = collect_and_transfer(label)
        output = torch.cat([output_input,label_input], 1)
        output = self.conv1(output)
        #print(self.conv1.weight.data)
        output = self.relu(output)
        output = output.view(45, 168)
        output = self.Linear1(output)

        return output

netG = Generator()
'''
x = np.loadtxt(input_file_path, delimiter=',', dtype=np.float32).T
y = np.loadtxt(output_file_path, delimiter=',', dtype=np.float32).T
x = torch.from_numpy(x)
y = torch.from_numpy(y)
print(netG(y, x))
'''
#netG.apply(weights_init)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(168, 84)
        self.linear2 = nn.Linear(84, 42)
        self.linear3 = nn.Linear(42, 42)
        self.linear4 = nn.Linear(42, 21)
        self.linear5 = nn.Linear(21, 10)
        self.linear6 = nn.Linear(10, 1)
        self.leakly = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.leakly(self.linear1(x))
        x = self.leakly(self.linear2(x))
        x = self.leakly(self.linear3(x))
        x = self.leakly(self.linear4(x))
        x = self.leakly(self.linear5(x))
        x = self.leakly(self.linear6(x))
        #x = self.sigmoid(x)
        return x

netD = Discriminator()
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD .parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.MSELoss()
loss = nn.MSELoss()
iters = 0
num_epochs = 500
G_losses, D_losses = [], []

for epoch in range(num_epochs):
    for i, data in enumerate(D_dataLoader):
        netD.zero_grad()
        real_cpu = data[0]
        #print(real_cpu.size())
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), 1, dtype=torch.float)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)

        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(45, 168)
        #print(noise)

        fake = netG(noise, data[1])

        label.fill_(0)
        fake = fake.view(45, 168)

        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)

        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = 0.5 * (errD_fake + errD_real)

        optimizerD.step()

        netG.zero_grad()
        label.fill_(1)
        output = netD(fake).view(-1)

        errG = loss(output, label)
        errG.backward()

        D_G_z2 = output.mean().item()

        optimizerG.step()

        if i % 50 == 0:
            print('[%d / %d][%d / %d]\tLoss_D: %.4f\tLoss_G : %.4f\tD(x) : %.4f\tD(G(z)) : %.4f / %.4f'
                  % (epoch, num_epochs, i, len(D_dataLoader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        iters += 1

torch.save(netG.state_dict(), "model1.pth")
print("Saved PyTorch Model State to model.pth")


model = Generator()
model.load_state_dict(torch.load("model1.pth"))

input_file_path = "dataset_4/dataset_4/input_4.csv"
output_file_path = "dataset_4/dataset_4/result_4.csv"
x = np.loadtxt(input_file_path, delimiter= ',', dtype=np.float32).T
y = np.loadtxt(output_file_path, delimiter= ',', dtype=np.float32)
y = y.T
mu = np.mean(y)
sigma = np.std(y)
ydata = (y - mu)/sigma
y = torch.from_numpy(ydata)
fake_noise = torch.randn(45, 168)
with torch.no_grad():
    pred = model(fake_noise, y).view(168, 45)
    pred = pred.cpu().detach().numpy()
    for i in range(len(pred)):
        for j in range(len(pred[0])):
            if pred[i][j] < 0:
                pred[i][j] = 0

print(pred)
np.savetxt('res1.csv', pred, delimiter=',', fmt='%.2f')










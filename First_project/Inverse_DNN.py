## based on result four to two
## total tandem network use in design

# import
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from torch import nn, optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# Initiaze the a set of outputs as inputs
class NewDataset(Dataset):
    def __init__(self, datapath, length):
        self.x_data = np.array([[random.uniform(65, 70), random.uniform(1500, 2500)] for i in range(length)])
        self.x_data = self.x_data.astype(np.float32)

        # Calculate the mu and sigma of the original data set
        xydata = np.loadtxt(datapath, delimiter= ',', dtype= np.float)
        self.len = xydata.shape[0]
        self.mu = []
        self.sigma = []
        for i in range(6):      # total amounts of inputs and outputs
            mu = np.mean(xydata[:, i])
            sigma = np.std(xydata[:, i])
            self.mu.append(mu)
            self.sigma.append(sigma)
            xydata[:, i] = (xydata[:, i] - mu) / sigma

        self.org = self.x_data
        self.x_data[:, 0] = (self.x_data[:, 0] - self.mu[4]) / self.sigma[4]
        self.x_data[:, 1] = (self.x_data[:, 1] - self.mu[5]) / self.sigma[5]
        self.x_data = torch.from_numpy(self.x_data).to(device)
        self.y_data = self.x_data
        self.len = length

    def __getitem__(self, index):
        return self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len

dataset = NewDataset('all_dataset.csv', 200)
train_loader = DataLoader(dataset = dataset, batch_size= dataset.len, shuffle= True, drop_last = True)

# Using pretrained forward DNN model needs to copy the word architecture first
class Model(nn.Module):
    def __init__(self):
        super(forward_model, self).__init__()
        # self.linear_relu_stack = nn.Sequential() not easy to read
        self.linear1 = nn.Linear(4, 16)
        self.linear2 = nn.Linear(16, 16)
        self.linear3 = nn.Linear(16, 16)
        self.linear4 = nn.Linear(16, 16)
        self.linear5 = nn.Linear(16, 2)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.linear5(x)
        return x


class inverse_model(nn.Module):
    def __init__(self, model):
        super(inverse_model, self).__init__()
        self.model = model
        for p in self.parameters():
            p.requires_grad = False
        self.linear1 = nn.Linear(2, 8)
        self.linear2 = nn.Linear(8, 8)
        self.linear3 = nn.Linear(8, 4)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        a, b, c, d = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        d = 10 + 80 * d
        c = (d - 10) * c
        c = (c - dataset.mu[2]) / dataset.sigma[2]
        d = (d - dataset.mu[3]) / dataset.sigma[3]
        x = torch.stack([a, b, c, d], 1)

        mid = x
        x = self.model(x)
        return x, mid


pretrained_foward_model = torch.load('fmodel.pt')
Inverse_model = inverse_model(pretrained_foward_model).to(device)

loss_fn = nn.MSELoss()
optim = optim.SGD(Inverse_model.parameters(), lr=0.05)

train_loss_record = []
def train(epoch):
    for batch_idx, (x, y) in enumerate(train_loader, 0):
        inputs, labels = x, y
        y_pred, _ = Inverse_model(inputs)

        loss = loss_fn(y_pred, labels)

        optim.zero_grad()
        loss.backward()

        optim.step()
    if epoch % 1000 == 0:
        print(loss.item())
        train_loss_record.append(loss.item())


total_epoch = 10000

for epoch in range(total_epoch):
    train(epoch)


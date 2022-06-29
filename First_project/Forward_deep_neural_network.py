# based on result two to four

# import
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from torch import nn, optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


# Preparing data for training with DataLoaders
class OrigDataset(Dataset):
    def __init__(self, datapath):
        xydata = np.loadtxt(datapath, delimiter=',', dtype=np.float32)
        self.len = xydata.shape[0]

        for i in range(6):      # six is the numbers of input + output
            mu = np.mean(xydata[:, i])
            sigma = np.std(xydata[:, i])
            xydata[:, i] = (xydata[:, i] - mu) / sigma

        self.x_data = torch.from_numpy(xydata[:, 0:4]).to(device)
        self.y_data = torch.from_numpy(xydata[:, 4:6]).to(device)

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len

Train_dataset = OrigDataset('Original_data.csv')
Val_dataset = OrigDataset('Val_data.csv')

Train_DataLoader = DataLoader(dataset = Train_dataset, batch_size= Train_dataset.len, shuffle=True, drop_last=True)
Val_DataLoader = DataLoader(dataset = Val_dataset, batch_size= Val_dataset.len, shuffle = True, drop_last=True)

# Create forward mdoel
class forward_model(nn.Module):
    def __init__(self):
        super(forward_model, self).__init__()
        # self.linear_relu_stack = nn.Sequential() not easy to read
        self.linear1 = nn.Linear(4, 16)
        self.linear2 = nn.Linear(16, 16)
        self.linear3 = nn.Linear(16, 16)
        self.linear4 = nn.Linear(16, 16)
        self.linear5 = nn.Linear(16, 2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.linear5(x)
        return x

Forward_model = forward_model().to(device)
loss_fn = nn.MSELoss()
optim = optim.SGD(Forward_model.parameters(), lr = 0.05)

train_loss_record = []
test_loss_record = []

# Train
def train(epoch):
    for batch, (x, y) in enumerate(Train_DataLoader, 0):
        inputs, labels = x, y
        y_pred = Forward_model(inputs)

        loss = loss_fn(y_pred, labels)

        # Backprop
        optim.zero_grad()
        loss.backward()

        optim.step()

    if epoch % 1000 == 0:
        print(loss.item())
        train_loss_record.append(loss.item())

def test(epoch):
    for batch, (x, y) in enumerate(Val_DataLoader, 0):
        inputs, labels = x, y
        with torch.no_grad():
            y_pred = Forward_model(inputs)
            loss = loss_fn(y_pred, labels)

    if epoch % 1000 == 0:
        print(loss.item())
        test_loss_record.append(loss.item())

total_epoch = 10000

for epoch in range(total_epoch):
    train(epoch)
    test(epoch)

# Save the pretrained network for future use
torch.save(Forward_model, 'fmodel.pt')













import torch
import torch.nn as nn
from torch.autograd import Variable

from models.fkopt import fkopt_net
from data.fkopt_data import FkoptDataset

learning_rate = 0.0001
num_joint = 10
batch_size = 1024
training_epochs = 500
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():
    model = fkopt_net(num_joint).to(device)
    
    loss = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = FkoptDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(training_epochs):
        for X, Y in dataloader:
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = model(X)
            cost = loss(hypothesis[:, -1], Y)
            cost.backward()
            optimizer.step()

        print(Y[0])
        print(hypothesis[:, -1][0])

        # run model and save image for this epoch
        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, cost))

def main():
    train()

if __name__ == '__main__':
    main()
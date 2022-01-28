import json
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.fkopt import fkopt_net
from data.fkopt_data import FkoptDataset
import time

learning_rate = 0.0001
num_joint = 10
batch_size = 1024
training_epochs = 1000
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SAVE_PATH = './checkpoints/fkopt.pt'

test_data = torch.tensor([[0.5, 0.5, 0.5]]).to(device)
test_results = []

def train():
    model = fkopt_net(num_joint).to(device)
    
    loss = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = FkoptDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    start = time.time()
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

        # test data observation time by time
        test_results.append(model(test_data).squeeze().to('cpu').detach().numpy().tolist())

    # save test data
    with open("images/data.json", "w") as json_file:
        json.dump(test_results, json_file)

    torch.save(model.state_dict(), SAVE_PATH)
    end = time.time()
    print('[Elapsed Time] : {}'.format(end - start))

def main():
    train()

if __name__ == '__main__':
    main()
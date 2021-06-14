import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(
        	self.num_layers, 
        	x.size(0), 
        	self.hidden_dim
        	).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

class Call_Strike_Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(32)
        self.input = nn.Linear(3, 32)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, inputs):
        x = nn.functional.relu(self.bn1(self.input(inputs)))
        x = nn.functional.relu(self.bn1(self.fc1(x)))
        x = nn.functional.relu(self.bn1(self.fc2(x)))
        x = nn.functional.relu(self.bn1(self.fc3(x)))
        return self.out(x)


def train_rnn(model, train_features, train_targets, num_epochs=100, threshold=1e-6, criterion=None, optimiser=None):
    """
    Performs training on a model given training data
    Used for recurrent NNs
    """

    hist = []
    criterion = nn.MSELoss(reduction='mean') if criterion == None else criterion
    optimiser = optim.Adam(model.parameters(), lr=0.01) if optimiser == None else optimiser
    prev_loss = 1
    loss_delta = 1
    epoch = 0

    print("\nBegin Training:")

    while loss_delta > threshold and epoch < num_epochs:

        train_target_pred = model(train_features)
        loss = criterion(train_target_pred, train_targets)
        loss_delta = abs(prev_loss - loss.item())
        prev_loss = loss.item()

        print("Epoch ", epoch, "MSE: ", loss.item())
        hist.append(loss.item())
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        epoch += 1

    print("Training Complete.\n")
    return np.array(hist)

def train_dnn(model, train_features, train_targets, num_epochs=100, threshold=1e-5, criterion=None, optimiser=None):
    """
    Performs training on a model given training data
    Used for Deep NNs (different criterion)
    """
    raise NotImplementedError



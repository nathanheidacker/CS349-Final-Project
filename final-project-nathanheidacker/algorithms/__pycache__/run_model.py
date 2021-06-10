import numpy as np
import pandas as pd
import torch

def train(model):
	"""
	Performs training on a model given training data
	"""

	hist = np.zeros(num_epochs)
	lstm = []

	criterion = torch.nn.MSELoss(reduction='mean')
	optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

	for t in range(num_epochs):
    	y_train_pred = model(x_train)
    	loss = criterion(y_train_pred, y_train)
    	print("Epoch ", t, "MSE: ", loss.item())
    	hist[t] = loss.item()
    	optimiser.zero_grad()
    	loss.backward()
    	optimiser.step()

    print("Training Complete.")
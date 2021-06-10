import numpy as np
import pandas as pd
import torch

def train_rnn(model, train_features, train_targets, num_epochs=100, threshold=1e-6, criterion=None, optimiser=None):
	"""
	Performs training on a model given training data
	Used for recurrent NNs
	"""

	hist = []
	criterion = torch.nn.MSELoss(reduction='mean') if criterion == None else criterion
	optimiser = torch.optim.Adam(model.parameters(), lr=0.01) if optimiser == None else optimiser
	prev_loss = 1
	loss_delta = 1
	epoch = 0

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

	print("Training Complete.")
	return np.array(hist)

def train_dnn(model, train_features, train_targets, num_epochs=100, threshold=1e-5, criterion=None, optimiser=None):
	"""
	Performs training on a model given training data
	Used for Deep NNs (different criterion)
	"""
	raise NotImplementedError
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def split_data(stock, lookback, split):
	"""
	splits stock data based on a 'lookback' period,
	or how many trading days into the past should be
	considered
	"""
	data_raw = stock.to_numpy()
	data = []

	for index in range(len(data_raw) - lookback):
		data.append(data_raw[index: index + lookback])

	data = np.array(data)
	test_set_size = int(np.round(split*data.shape[0]))
	train_set_size = data.shape[0] - test_set_size

	x_train = data[:train_set_size, :-1, :]
	y_train = data[:train_set_size, -1, :]

	x_test = data[train_set_size:, :-1]
	y_test = data[train_set_size:, :-1, :]

	return [x_train, y_train, x_test, y_test]


def spy_price_data():
	"""
	Returns spy closing price data
	"""
	data = pd.read_csv("data/spy.csv")
	return data


def load_RNN_spy_data(lookback=20, split=0.2):
	"""
	Returns spy data formatted for NN training
	"""
	data = pd.read_csv("data/spy.csv")

	price = data[['Close']]
	scaler = MinMaxScaler(feature_range=(-1, 1))
	price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))

	x_train, y_train, x_test, y_test = split_data(price, lookback, split)

	x_train = torch.from_numpy(x_train).type(torch.Tensor)
	x_test = torch.from_numpy(x_test).type(torch.Tensor)
	y_train = torch.from_numpy(y_train).type(torch.Tensor)
	y_test = torch.from_numpy(y_test).type(torch.Tensor)

	return x_train, x_test, y_train, y_test, scaler
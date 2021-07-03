import re
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

def isoformat_dates(data, date_format=None):
	"""
	Requires a non-empty dataset, returns a minimum viable dataset formatted
	to function properly with finance tools.
	"""
	if "DATE" not in data.columns:
		raise ValueError("Asset dataset not initialized, requires date information")

	if date_format:
		try:
			x = format_date(date_format)
		except:
			verboseprint

	standard_date_formats = ["YYYY/DD/MM", "DD/MM/YYYY"]

	def format_date(template):
		"""
		Given a template, returns a function that can turn the given template
		into an isoformatted datestring compatible with the datetime package.
		Returned function will take input strings of the format 'template',
		and return strings of the form "YYYY-MM-DD"
		"""
		Y = []
		M = []
		D = []
		for i in range(len(template)):
			if template[i] in ['y', 'Y']:
				Y.append(i)
			elif template[i] in ['m', 'M']:
				M.append(i)
			elif template[i] in ['d', "D"]:
				D.append(i)

		def formatter(datestring):
			year = "".join([datestring[i] for i in Y])
			month = "".join([datestring[i] for i in M])
			day = "".join([datestring[i] for i in D])
			formatted = "-".join([year, month, day])
			return formatted

		return formatter

	formatter = format_date("YYYY/DD/MM")

	return formatter(data)

	# Check x


def format_asset_pricedata(data):
	pass


def format_asset_dataset(data):
	pass


def asset_dataset(data=None):
	"""
	Creates a dataset intended for use by asset objects
	"""

	# Dataset Initialization
	dataset = pd.Dataset()

	# Checking that data is an acceptable type
	if not isinstance(data, [pd.DataFrame, np.array, str, type(None)]):
		raise ValueError("Asset Dataset could not be initialized, {} is an invalid input".format(type(data)))

	# Initializing dataset based on type
	if type(data) == str:
		dataset = pd.read_csv(data)

	elif type(data) == np.array:
		dataset = pd.DataFrame(data)

	elif data == None:
		dataset = pd.DataFrame()

	elif type(data) == pd.DataFrame:
		dataset = data

	else:
		try:
			self.data = pd.DataFrame(data)
		except:
			raise ValueError("Data could not be initialized")

	return format_asset_dataset(dataset) if not dataset.empty else dataset


def stock_dataset(name, data=None):
	"""
	Creates a dataset intended for use by stock objects
	"""

	# Dataset initialization
	dataset = asset_dataset(data)

	if self.dataset.empty:
		pass



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
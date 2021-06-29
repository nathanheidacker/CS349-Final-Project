import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime
import time
from data import load_data
from algorithms import models, finance

# TESTING

# MODEL TRAINING

lookback = 20
split = 0.1
threshold = 5e-6

x_train, x_test, y_train, y_test, scaler = load_data.load_RNN_spy_data(lookback, split)

input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100

lstm = models.LSTM(
	input_dim=input_dim,
	hidden_dim=hidden_dim,
	output_dim=output_dim,
	num_layers=num_layers
	)

start_time = time.time()

lstm_hist = models.train_rnn(lstm, x_train, y_train, num_epochs, threshold)
    
training_time = time.time() - start_time
print("Training time: {} seconds".format(round(training_time, 2)))

gru = models.GRU(
	input_dim=input_dim,
	hidden_dim=hidden_dim,
	output_dim=output_dim,
	num_layers=num_layers
	)

start_time = time.time()

gru_hist = models.train_rnn(gru, x_train, y_train, num_epochs, threshold)
    
training_time = time.time() - start_time
print("Training time: {} seconds".format(round(training_time, 2)))

x, _, y, _, _ = load_data.load_RNN_spy_data(split=0.0)

# The part of the data that we want to visualize.
# In this case, we only want to visualize data that
# the network has not trained on. In other words,
# performance on novel data.
zoom = split

data = load_data.spy_price_data()
price = data[["Close"]]
dates = data[["Date"]].to_numpy().flatten()
dates = dates[round(dates.shape[0] * (1 - zoom)):]
date_pos = range(len(dates))[::800]
tick_dates = dates[::800]

#closes = data[["Close"]].to_numpy().flatten()
closes = scaler.inverse_transform(x.detach().numpy()[:, 0, :])
closes = closes[round(closes.shape[0] * (1 - zoom)):]
lstm_closes = scaler.inverse_transform(lstm(x).detach().numpy())
lstm_closes = lstm_closes[round(lstm_closes.shape[0] * (1 - zoom)):]
gru_closes = scaler.inverse_transform(gru(x).detach().numpy())
gru_closes = gru_closes[round(gru_closes.shape[0] * (1 - zoom)):]

# CREATING PLOTS FOR PREDICTION PERFORMANCE

plt.title("LSTM and GRU Performance on SPY Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.plot(closes, label="Real Closing Price")
plt.plot(lstm_closes, label="LSTM Predictions")
plt.plot(gru_closes, label="GRU Predictions")
plt.xticks(date_pos, tick_dates)
plt.legend()
plt.show()

plt.title("LSTM and GRU Loss over Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(lstm_hist, label="LSTM Loss")
plt.plot(gru_hist, label="GRU Loss")
plt.legend()
plt.show()

# COVERED CALL GRAPHS

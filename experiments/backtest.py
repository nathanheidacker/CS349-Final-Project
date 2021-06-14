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

x_train, x_test, y_train, y_test, scaler = load_data.load_spy_data(20)

input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 1

lstm = models.LSTM(
	input_dim=input_dim,
	hidden_dim=hidden_dim,
	output_dim=output_dim,
	num_layers=num_layers
	)

start_time = time.time()

lstm_hist = run_model.train_rnn(lstm, x_train, y_train, num_epochs)
print(lstm_hist)
    
training_time = time.time() - start_time
print("Training time: {} seconds".format(round(training_time, 2)))

gru = models.GRU(
	input_dim=input_dim,
	hidden_dim=hidden_dim,
	output_dim=output_dim,
	num_layers=num_layers
	)

start_time = time.time()

gru_hist = run_model.train_rnn(gru, x_train, y_train, num_epochs)
    
training_time = time.time() - start_time
print("Training time: {} seconds".format(round(training_time, 2)))

x, _, y, _, _ = load_data.load_spy_data(split=0)

data = load_data.spy_price_data()
price = data[["Close"]]
dates = data[["Date"]].to_numpy().flatten()
date_pos = range(len(dates))[::800]
dates = dates[::800]
closes = data[["Close"]].to_numpy().flatten()
lstm_closes = scaler.inverse_transform(lstm(x).detach().numpy())
gru_closes = scaler.inverse_transform(gru(x).detach().numpy())

# CREATING PLOTS FOR PREDICTION PERFORMANCE

plt.title("LSTM and GRU Performance on SPY Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.plot(closes, label="Real Closing Price")
plt.plot(lstm_closes, label="LSTM Predictions")
plt.plot(gru_closes, label="GRU Predictions")
plt.xticks(date_pos, dates)
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

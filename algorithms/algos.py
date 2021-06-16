import numpy as np
import pandas as pd
import torch
import datetime
from algorithms import finance, models
from data import load_data

def spy_covered_calls(portfolio, start_date=None, end_date=None):
	"""
	Performs the SPY covered call strategy on a portfolio, given a
	start date. If no end date is given, the algorithm will run until
	the present day

	Arguments:
		portfolio: a finance.portfolio object on which the algorithm
			is performed
		start_date: a datetime.date object that marks the starting
			point of the algorithm. Can also be passed in as an 
			iso-formatted date string: "YYYY-MM-DD"
		end_date: a datetime.date object that marks the ending
			point of the algorithm. Can also be passed in as an 
			iso-formatted date string: "YYYY-MM-DD". Optional

	Returns:
		modifies the portfolio passed into portfolio
		returns alpha?
	"""

	# Convert all start_date inputs to datetime objects, grab
	# portfolio start date if none is given
	if start_date:
		if type(start_date) == str:
			start_date = datetime.date.fromisoformat(start_date)
	else:
		start_date = list(portfolio.history.keys())[0]
		start_date = datetime.date.fromisoformat(start_date)

	# Convert all end_date inputs to datetime objects, iterate
	# until today's date if none is given
	if end_date:
		if type(end_date) == str:
			end_date = datetime.date.fromisoformat(end_date)
	else:
		end_date = datetime.date.today()

	# Variable initialization, getting data
	data = pd.read_csv("data/spy.csv")
	current_date = start_date

	# Ending criteria, end_date is the last date we process
	while current_date <= end_date:

		# Grabbing row values that correspond to current_date
		current_data = data.loc[data["Date"] == current_date.isoformat()]
		if not today.empty:
			open_price = current_data.iloc[0]["Open"]
			available_cash = portfolio.positions["cash"]
			spy_today = finance.Stock("SPY", open_price, 100)



		# Trading day is over, increment day
		current_date += datetime.timedelta(days=1)



def spy_covered_calls_NN(portfolio, start_date=None, end_date=None):
	"""
	Performs the SPY covered call strategy on a portfolio, given a
	start datae. If no end date is give, the algorithm will run until
	the present day.

	This version implements trained neural networks for the
	determination of future SPY value, and uses those predicted
	spot values in the determination of the optimal call strike
	price to collect maximum premium

	Arguments:
		portfolio: a finance.portfolio object on which the algorithm
			is performed
		start_date: a datetime.date object that marks the starting
			point of the algorithm. Can also be passed in as an 
			iso-formatted date string: "YYYY-MM-DD"
		end_date: a datetime.date object that marks the ending
			point of the algorithm. Can also be passed in as an 
			iso-formatted date string: "YYYY-MM-DD". Optional

	Returns:
		portfolio: returns a new portfolio on which the algorithm has
			been performed from start_date to end_date.
	"""
	return portfolio
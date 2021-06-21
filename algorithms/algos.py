import numpy as np
import pandas as pd
import torch
import datetime
from algorithms import finance, models
import algorithms.algorithm_helpers as helpers
from data import load_data

def spy_covered_calls(portfolio, start_date=None, end_date=None):
	"""
	Performs the SPY covered call strategy on a portfolio, given a
	start date. If no end date is given, the algorithm will run until
	the present day

	Arguments:
		portfolio {Portfolio}: 
			a finance.portfolio object on which the algorithm
			is performed

		start_date {Date / Timedelta / Str}: 
			Date: A datetime.date object that acts as the end
			point of the algorithm

			Timedelta: A datetime.timedelta object that acts as
			the time to perform the algorithm from the start date

			Str: A iso-formatted date string ("YYYY-MM-DD") that
			acts as the end point of the algorithm

		end_date {Date / Timedelta / Str}: 
			Date: A datetime.date object that acts as the end
			point of the algorithm

			Timedelta: A datetime.timedelta object that acts as
			the time to perform the algorithm from the start date

			Str: A iso-formatted date string ("YYYY-MM-DD") that
			acts as the end point of the algorithm

			Note - end_date is INCLUSIVE, algorithm will continue
			until the portfolio.date EXCEEDS the end_date

	Returns:
		Modifies the portfolio passed into the input

	"""

	# Variable initialization, getting data
	_, end_date = helpers.initialize(portfolio, start_date, end_date)
	data = pd.read_csv("data/spy.csv")

	# Ending criteria, end_date is the last date we process
	while portfolio.date < end_date:

		# Grabbing row values that correspond to current_date
		current_data = data.loc[data["Date"] == portfolio.date.isoformat()]

		# Perform algorithm when data is available
		if not current_data.empty:
			open_price = current_data.iloc[0]["Open"]
			available_cash = portfolio.positions["cash"]

			spy = finance.Stock(open_price, 100, "SPY")



		# Trading day is over, increment day
		portfolio.next_day()



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
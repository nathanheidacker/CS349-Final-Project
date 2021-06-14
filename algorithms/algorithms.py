import numpy as np
import pandas as pd
import torch
import finance, models
from data import load_data

def spy_covered_calls(portfolio, start_date, end_date=None):
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
		portfolio: returns a new portfolio on which the algorithm has
			been performed from start_date to end_date.
	"""

def spy_covered_calls_NN(portfolio, start_date, end_date=None, )
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

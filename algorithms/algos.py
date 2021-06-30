import math
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
	# The spy options chain schedule, integer values correspond to days
	# These values determine the day of expiry for a short sold call given
	# the weekday it is being sold on
	expiry_schedule = {
		0: 2,
		1: 1,
		2: 0,
		3: 1,
		4: 0,
	}

	# Initialize portfolio
	start_date, end_date = helpers.initialize(portfolio, start_date, end_date)
	spy = finance.Stock("SPY", "data/spy.csv", start_date, dividend_yield=0.015, initial_key="Open")

	# Ending criteria, end_date is the last date we process
	while portfolio.date <= end_date:

		# Market is closed on weekends
		if portfolio.date.weekday() < 5:

			# Determine the number of shares we can buy based on our liquid assets
			stock_purchase_volume = math.floor(portfolio.liquid().value / spy.price / 100) * 100
			if stock_purchase_volume > 0:
				portfolio.buy(spy, stock_purchase_volume)

			# Calculate current spy volume and spy call volume
			current_spy_volume = getattr(portfolio.position(spy), "volume", 0)
			current_spy_call_volume = sum([position.volume for position in portfolio.positions.get("call", []) if position.asset.name == "SPY"])

			# The call to be sold
			basis = getattr(portfolio.position(spy), "basis", None)
			spy_call = finance.Call(spy, max(math.floor(spy.price) + 2, basis), expiry_schedule[portfolio.date.weekday()])

			# Determine number of calls that can be sold out
			call_short_volume = (current_spy_volume // 100) - current_spy_call_volume
			if call_short_volume > 0:
				portfolio.short(spy_call, call_short_volume)

			# Evaluate positions at the end of the day
			# Expired calls will be assigned or expire worthless
			portfolio.valuate("Close")

		# Go to the next trading day unless today was the last day the algorithm
		# was to run
		if portfolio.date == end_date:
			break

		else:
			portfolio.next_day("Open")


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
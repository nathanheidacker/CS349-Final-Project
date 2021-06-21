from algorithms import finance
from copy import deepcopy
import datetime

def initialize(portfolio, start_date=None, end_date=None):
	"""
	***MUST BE CALLED AT THE BEGINNING OF A NEW ALGORITHM***

	Handles the necessary variable initialization of a new algorithm.
	Accepts a datetime.date, datetime.timedelta, or str object for
	both start and end date inputs, returning the correct datetime.date
	object corresponding to each.

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

	Returns:

		Updates the portfolio.date of the portfolio passed in
		depending on the start date given

		start_date {datetime.date}: 
			The starting date of the algorithm

		end_date {datetime.date}: 
			The ending date of the algorithm

	"""
	# When start_date input is received
	if start_date:

		# Handle Str start_date inputs
		if type(start_date) == str:
			start_date = datetime.date.fromisoformat(start_date)

		# Handle datetime.timedelta start_date inputs
		elif type(start_date) == datetime.timedelta:
			start_date = portfolio.date + start_date

		# The starting date of the algoritm can not be before
		# the current date of the portfolio
		if start_date < portfolio.date:
			raise ValueError("Provided start date ({}) is before\
				the portfolio's current date ({})".format(
					start_date,
					portfolio_start))

		# Updating the portfolio.date to new starting date
		portfolio.date = start_date

	else:

		# If no start_date is given, we use portfolio.date
		start_date = portfolio.date

	# When end_date input is received
	if end_date:

		# Handle Str end_date inputs
		if type(end_date) == str:
			end_date = datetime.date.fromisoformat(end_date)

		# Handle datetime.timedelta end_date inputs
		elif type(end_date) == datetime.timedelta:
			end_date = portfolio.date + end_date

	else:

		# If no end_date is given, we use today
		end_date = datetime.date.today()

	# Make sure that end_date is not before start_date
	if end_date < start_date:
		raise ValueError("End date {} is before start date {}"\
			.format(end_date, start_date))

	return start_date, end_date



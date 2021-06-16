import math
import datetime
import numpy as np
import datetime
import csv
import matplotlib.pyplot as plt
from copy import deepcopy

def cdf(x):
	"""
	Cumulative distribution function for the standard
	normal distribution
	"""
	return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def black_scholes(spot, strike, rfr, ttm, vol):
	"""
	An implementation of the black scholes equation taken
	from wikipedia. Used in the calculation of call option
	premiums

	Arguments:
		spot: The current spot price of the underlying asset
		strike: The strike price of the call option
		rfr: The current risk free rate
		ttm: The time to maturity of the call option
		vol: The current measure of historical volatility of
			 the underlying asset

	Returns:
		premium: The call option's current value (premium)
	
	d1p1 = (math.log(spot / strike) + ((rfr + ((vol ^ 2) / 2)) * ttm))
	d1p2 = vol * ttm^0.5
	d1 = d1p1 / d1p2
	d2 = d1 - d1p2
	premium = cdf(d1) * spot
	premium -= cdf(d2) * strike * math.e ^ (-1 * rfr * ttm)
	return premium
	"""
	d1 = (math.log(spot / strike) + ((rfr + ((vol * vol) / 2)) * ttm)) / (vol * math.sqrt(ttm))
	d2 = d1 - (vol * math.sqrt(ttm))
	call_premium = (cdf(d1) * spot) - (cdf(d2) * strike * math.pow(math.e, -1 * (rfr * ttm)))
	return call_premium

class Valuation:
	"""
	A valuation of an asset, tied to a specific date.
	Valuations do not require a date in some cases, such as:
		* Initialized asset value
		* Constant value over time
		* Unknown valuation date

	"""
	def __init__(self, price, date=None):
		self.price = price
		self.date = date

class Asset:
	"""
	Base class for financial assets.
	Carries some basic information shared among all assets
	"""
	def __init__(self, price, volume):
		self.asset_type = None
		self.basis = price
		self.volume = volume
		self.spot = Valuation(price)


	def __repr__(self):
		return str(self)

	def auto_valuate(self):
		pass

class Currency(Asset):
	"""
	Base class for currency assets
	"""
	def __init__(self, price, volume, name):
		super(Currency, self).__init__(price, volume)
		self.asset_type = name

	def __str__(self):
		return "({} x {} @ ${})".format(self.asset_type, self.volume, self.basis)

	def __add__(self, other):
		if self.asset_type == other.asset_type:
			self_total = self.basis * self.volume
			other_total = other.basis * other.volume
			total = self_total + other_total
			total_volume = self.volume + other.volume
			new_basis = total / total_volume
			self.basis = new_basis
			self.volume = total_volume
			return self
		else:
			raise ValueError(
				"Can not add currencies \"{}\" and \"{}\", must be the same type"\
				.format(self.asset_type, other.asset_type))

	def __sub__(self, other):
		if self.asset_type == other.asset_type:
			self_total = self.basis * self.volume
			other_total = other.basis * other.volume
			total = self_total - other_total
			total_volume = self.volume - other.volume
			if total_volume == 0:
				if self.asset_type == "cash":
					return Cash(0)
				else:
					return Currency(self.basis, 0, self.asset_type)
			else:
				new_basis = total / total_volume
				self.basis = new_basis
				self.volume = total_volume
				return self
		else:
			raise ValueError(
				"Can not subtract currencies \"{}\" and \"{}\", must be the same type"\
				.format(self.asset_type, other.asset_type))


class Cash(Currency):
	"""
	Class for the United States Dollar
	"""
	def __init__(self, amount):
		super(Cash, self).__init__(1, amount, "cash")

	def __str__(self):
		return "(${})".format(self.volume)

	def __add__(self, other):
		if type(other) in [int, float]:
			self.volume += other
			return self
		else:
			return super(Cash, self).__add__(other)

	def __sub__(self, other):
		if type(other) in [int, float]:
			self.volume -= other
			return self
		else:
			return super(Cash, self).__sub__(other)

class Stock(Asset):
	"""
	Class representing a financial security
	"""
	def __init__(self, price, volume, ticker):
		super(Stock, self).__init__(price, volume)
		self.asset_type = "stock"
		self.name = ticker

	def __str__(self):
		return "({} {} shares @ ${})".format(
			self.volume, 
			self.name, 
			self.spot.price
			)

	def __add__(self, other):
		if self.name == other.name:
			self_total = self.basis * self.volume
			other_total = other.basis * other.volume
			total_volume = self.volume + other.volume
			new_basis = (self_total + other_total) / total_volume
			return Stock(self.name, new_basis, total_volume)
		else:
			return ValueError("Can not combine two stock positions \
				in different companies")

	def __sub__(self, other):
		if self.name == other.name:
			self_total = self.basis 

	def valuate(self, price, date=None):
		"""
		updates the current spot price of the asset
		"""
		self.spot = Valuation(price, date)

	def auto_valuate(self):
		raise NotImplementedError


class Option(Asset):
	"""
	Base class for financial derivate type options
	"""
	def __init__(self, premium, volume, ticker, strike, expiry):
		super(Option, self).__init__(premium, volume)
		self.name = ticker
		self.strike = strike
		self.expiry = expiry


class Call(Option):
	"""
	Class representing american call options
	"""
	def __init__(self, premium, volume, ticker, strike, expiry):
		super(Call, self).__init__(premium, volume, ticker, strike, expiry)
		self.asset_type = "call"

	def __str__(self):
		return "({} ${}C {} ({}))".format(
			self.name,
			self.strike,
			self.expiry.isoformat(),
			self.volume
			)

	def valuate(self, value, date):
		"""
		updates the spot price of the asset
		"""
		if self.expiry > date:
			self.spot = (0, date)
		else:
			self.spot = (value, date)

	def expire(self):
		raise NotImplementedError

	def exercise(self):
		raise NotImplementedError

	def black_scholes_valuate(self):
		raise NotImplementedError

class Short_Call(Call):
	"""
	Class representing a call that has been sold
	"""

	def __init__(self, premium, volume, ticker, strike, expiry, collateral):
		raise NotImplementedError
		self.asset_type = "call"
		self.collateral = collateral

class Put(Option):
	"""
	Class representing american put options
	"""
	def __init__(self, premium, volume, ticker, strike, expiry):
		super(Call, self).__init__(premium, volume, ticker, strike, expiry)
		self.asset_type = "put"

	def __str__(self):
		return "({} ${}P {} ({}))".format(
			self.name,
			self.strike,
			self.expiry.isoformat(),
			self.volume
			)

	def valuate(self, value, date):
		"""
		updates the spot price of the asset
		"""
		if self.expiry > date:
			self.spot = (0, date)
		else:
			self.spot = (value, date)


class Portfolio:
	"""
	Class representing a portfolio which carries a set of financial
	positions, and a history of those positions across time.

	Arguments:
		initial {float / int}: A number value representing the initial
			capital in the portfolio, or the principal investment
		start_date {datetime.date / string}: A datetime.date object
			which represents the starting point of the portfolio.
			Can also be passed in as a date isoformat string
			"YYYY-MM-DD"
		risk_free_rate {float}: A float value indicating the annual
			growth of capital that can be achieved without assuming
			any risk. Default value is the average annual index
			return (8%)
		margin {float}: A float value that acts as a multiplier of
			the total account value to determine how much
	"""
	def __init__(self, initial, start_date, risk_free_rate=0.08, margin=None):
		self.maintenance_requirement = margin
		self.risk_free_rate = risk_free_rate
		if type(start_date) == str:
			start_date = datetime.date.fromisoformat(start_date)
		self.current_date = start_date
		self.positions = {"cash": [Cash(initial)]}
		self.history = {start_date.isoformat(): deepcopy(self.positions)}
		self.valid_assets = ["cash", "stock", "calls"]

	def __str__(self):
		result = ""
		for asset_type, positions in self.positions.items():
			pos_string = ""
			for position in positions:
				pos_string += "{}, ".format(position)
			result += "{}: {}\n".format(asset_type, pos_string[:-2])
		if result:
			return result[:-1]
		else:
			return "Empty portfolio"

	def reset(self):
		"""
		resets the portfolio to its initial conditions
		"""
		start, initial = self.history.items()[0]
		self.positions = self.initial
		self.history = {start: initial}

	def get_all_assets(self, positions=None):
		"""
		Returns a list of all assets
		"""

		positions = positions if positions else self.positions

		assets = []
		for asset_type in list(positions.keys()):
			for asset in positions[asset_type]:
				assets.append(asset)

		return assets

	def next_day(self):
		"""
		Iterate the portfolio's current day to the next available day
		"""
		self.current_date += datetime.timedelta(days=1)

	def invest(self, amount, date):
		"""
		used to increase principal investment into the
		portfolio by directly depositing cash
		"""
		self.positions["cash"] += amount
		self.history[date.isoformat()] = deepcopy(self.positions)

	def buy(self, position, date):
		"""
		used to enter an asset position
		"""
		if position.asset_type not in self.valid_assets:
			raise ValueError("Not a viable asset type")
		else:
			self.positions[position.asset_type].append(position)
			self.positions["cash"] -= position.basis * position.volume
			self.history[date.isoformat()] = deepcopy(self.positions)

	def sell(self, position, date):
		"""
		used to exit an asset position
		"""
		if type(date) == str:
			date = datetime.date.fromisoformat(date)

		all_assets = self.get_all_assets()
		if position.asset_type not in self.valid_assets:
			raise ValueError("Not a viable asset type")
		elif position not in all_assets:
			raise ValueError("Portfolio does not own this position")
		else:
			position = next((x for x in all_assets if x == position), None)
			self.positions["cash"] += position.spot.price * position.volume
			self.positions[position.asset_type].remove(position)
			self.history[date.isoformat()] = deepcopy(self.positions)

	def short(self, position, date):
		"""
		used to short a position
		"""
		raise NotImplementedError


	def valuate(self, date):
		"""
		automatically updates the values of all positions
		currently held.
		"""
		raise NotImplementedError

	def _value(self, date):
		"""
		Value helper, not for outside use.
		"""

		# all_assets must be the assets from this date
		all_assets = self.get_all_assets(self.history[date])

		x = self.history[date]
		total = 0
		for position in all_assets:
			if position.spot.date != date:
				position.auto_valuate()
			total += position.spot.price * position.volume
		return np.array([date, total])

	def value(self, date=None):
		"""
		Returns total portfolio value at a specified date.
		If the date is not specified, returns an np.array containing
		total portfolio value at each date.
		"""
		if date:
			return _value(date)[1]
		else:
			result = []
			for date in self.history:
				result.append(self._value(date))
			return np.array(result)

	def print_positions(self, positions=None, date=None):
		"""
		prints a formatted version of current portfolio
		positions. Positions and dates can be given to
		alter output.
		"""

		# Can print alternate positions given at a different date
		print_date = date if date else "CURRENT"
		positions = positions if positions else self.positions

		print("\n{} POSITIONS\n--------------------".format(print_date))
		"""
		for asset_type, assets in positions.items():
			if asset_type == "cash":
				print("{}: ${}".format(asset_type, assets))
			elif assets:
				result = "{}: ".format(asset_type)
				for i, asset in enumerate(assets):
					result += str(asset)
					if i < len(assets) - 1:
						result += ", "
				print(result)
		"""
		print_portfolio = Portfolio(0, datetime.date.today())
		print_portfolio.positions = positions
		print(str(print_portfolio))
		print("--------------------\n")


	def print_history(self):
		"""
		prints a formatted version of the portfolio history
		"""
		for date, positions in self.history.items():
			self.print_positions(positions, date)

	def show_history(self):
		"""
		Produces a matplotlib line graph of the current
		portfolio history.
		"""
		value_history =  self.value()
		x = value_history[:, 0]
		y = value_history[:, 1]
		plt.plot(x, y, label="Account Value")
		plt.xlabel("Date")
		plt.ylabel("Portfolio Value (USD)")
		plt.title("Portfolio Performance")
		plt.show()





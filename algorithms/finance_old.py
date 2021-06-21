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
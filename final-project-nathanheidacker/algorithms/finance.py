import math
import datetime
import numpy as np
import csv
from copy import deepcopy

def cdf(x):
	"""
	Cumulative distribution function for the standard
	normal distribution
	"""
	return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def getWeekday(date):
	info = date.split("/")
	d = datetime.date(int(info[2]), int(info[0]), int(info[1]))
	return d.weekday()

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

def algoDict(filepath, year):
    year = str(year)
    reader = csv.reader(open(filepath), delimiter=',')
    algoDict = {}
    for line in reader:
        if year in line[0]:
            algoDict[line[0]] = (line[1], line[2])
    return algoDict

def covered_calls(principal, security, year):

    #Initialization
    totalValue = principal
    portfolio = 0
    cash = totalValue - portfolio
    curr = Call(0, 0, 0)
    invested = False
    callSold = False
    assignments = 0
    data = algoDict(security, year)

    expiries = {0:2, 1:2, 2:4, 3:4, 4:0}

    #Algorithm Loop
    for date in data:
        price = float(data[date][0])
        closePrice = float(data[date][1])

        #Invest available money into shares
        if not invested:
            shares = int(round(totalValue / price, -2)) if round(totalValue / price, -2) < (totalValue / price) else int(round(totalValue / price, -2) - 100)
            portfolio = (shares * price, shares)
            cash = totalValue - portfolio[0]
            invested = True

        #Sell calls for shares available to cover
        if not callSold:
            contracts = portfolio[1] / 100
            premium = 0.7 * black_scholes(price, round((portfolio[0] / portfolio[1]) + 1), 0.08, 2/365, 0.12)
            totalValue += premium * contracts * 100
            cash += premium * contracts * 100
            curr = Call(round((portfolio[0] / portfolio[1]) + 1), expiries[getWeekday(date)], premium)
            callSold = True

        #Check if calls have been assigned
        if getWeekday(date) == curr.expiry:
            callSold = False
            if closePrice > curr.strike:
                assignments += 1
                cash += portfolio[1] * curr.strike
                portfolio = (0, 0)
                totalValue = cash
                invested = False

    return round(totalValue, 2), assignments


class Stock:
	def __init__(self, ticker, price, volume):
		self.asset_type = "stock"
		self.ticker = ticker
		self.basis = price
		self.spot = (price, None)
		self.volume = volume

	def __str__(self):
		return "({} {} shares @ ${})".format(
			self.volume, 
			self.ticker, 
			self.basis
			)

	def __repr__(self):
		return str(self)

	def __add__(self, other):
		if self.ticker == other.ticker:
			self_total = self.basis * self.volume
			other_total = other.basis * other.volume
			total_volume = self.volume + other.volume
			new_basis = (self_total + other_total) / total_volume
			return Stock(self.ticker, new_basis, total_volume)
		else:
			return ValueError("Can not combine two stock positions \
				in different companies")

	def value(self, value, date):
		"""
		updates the current spot price of the asset
		"""
		self.spot = (value, date)

class Call:
	def __init__(self, strike, expiry, premium):
		self.strike = strike
		self.premium = premium
		self.expiry = expiry

class Call2:
	def __init__(self, ticker, strike, expiry, premium, volume):
		self.asset_type = "call"
		self.ticker = ticker
		self.strike = strike
		self.expiry = expiry
		self.basis = premium
		self.spot = (premium, None)
		self.volume = volume

	def __str__(self):
		if self.volume > 1:
			return "({} ${} {} ({}))".format(
				self.ticker,
				self.strike,
				self.expiry.isoformat(),
				self.volume
				)
		else:
			return "({} ${} {})".format(
				self.ticker,
				self.strike,
				self.expiry.isoformat(),
				)

	def __repr__(self):
		return str(self)

	def value(self, value, date):
		"""
		updates the current spot price of the asset
		"""
		if self.expiry > date:
			self.spot = (0, date)
		else:
			self.spot = (value, date)


class Portfolio:
	def __init__(self, initial, start_date):
		self.positions = \
			{
			"cash": initial,
			"stock": [],
			"call": []
			}
		self.history = {start_date.isoformat(): self.positions}

	def __str__(self):
		return "Cash: {} \nStock: {} \nCalls: {} \n".format(
			self.positions["cash"],
			self.positions["stock"],
			self.positions["calls"]
			)

	def reset(self):
		"""
		resets the portfolio to its initial conditions
		"""
		start, initial = self.history.items()[0]
		self.positions = self.initial
		self.history = {start: initial}

	def get_all_assets(self):
		"""
		returns a list of all assets excluding cash
		"""
		result = []
		for asset_type in list(self.positions.keys())[1:]:
			for asset in self.positions[asset_type]:
				result.append(asset)

		return result

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
		if position.asset_type not in ["call", "stock"]:
			raise ValueError("Not a viable asset type")
		else:
			self.positions[position.asset_type].append(position)
			self.positions['cash'] -= position.basis * position.volume
			self.history[date.isoformat()] = deepcopy(self.positions)

	def sell(self, position, date):
		"""
		used to exit an asset position
		"""
		all_assets = self.get_all_assets()
		if position.asset_type not in ["call", "stock"]:
			raise ValueError("Not a viable asset type")
		elif position not in all_assets:
			raise ValueError("Portfolio does not own this position")
		else:
			position = next((x for x in all_assets if x == position), None)
			self.positions['cash'] += position.spot[0] * position.volume
			self.positions[position.asset_type].remove(position)
			self.history[date.isoformat()] = deepcopy(self.positions)

	def _value(self, date):
		"""
		value helper
		"""
		x = self.history[date.isoformat()]
		total = x["cash"]
		for position in x["stock"] + x["call"]:
			if position.spot[1] != date:
				position.value()
			total += position.spot
		return np.array([date.isoformat(), total])

	def value(self, date=None):
		"""
		Returns total portfolio value at a specified date.
		If the date is not specified, returns an np.array containing
		total portfolio value at each date
		"""
		if date:
			return _value(date)[1]
		else:
			result - []
			for date in self.history:
				result.append(_value(date))
			return np.array(result)



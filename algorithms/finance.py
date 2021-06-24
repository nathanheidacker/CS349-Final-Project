import math
import datetime
import numpy as np
import pandas as pd
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

class Asset:
	"""
	Class representing a financial asset. Dynamically tracks price
	asset price across time when paired with a dataset

	Arguments:
		asset_type {Str}:
			The type of asset being represented by the instantiation

		name {Str}:
			The name of the asset. In the case of financial securities,
			this is usually the ticker

		price {Int / Float}:
			A numeric value indicating the initialized price of this asset

		date {datetime.date / Str}:
			A datetime.date object representing the current date tracked
			by the asset. Can also be passed in as an isoformatted date
			string: "YYYY-MM-DD"

		data {pandas.dataframe}:
			A pandas dataframe associated with this asset object,
			used to track the asset price across time.
	"""

	def __init__(self, asset_type, name, price, date=None, data=None, fractional=True, viable_collateral=None):

		# Attribute intialization
		self.type = asset_type
		self.name = name
		self.price = None if price == None else float(price)
		self.date = datetime.date.fromisoformat(date) if type(date) == str else date
		self.fractional = fractional
		self.viable_collateral = viable_collateral if viable_collateral else []
		self.data = pd.DataFrame()

		# Making sure data is a dataframe object, otherwise object cant be initialized
		if type(data) == str:
			self.data = pd.read_csv(data)
		elif type(data) == np.array:
			self.data = pd.DataFrame(data)
		elif type(data) == type(None):
			self.data = pd.DataFrame()
		elif type(data) == pd.DataFrame:
			self.data = data
		else:
			try:
				self.data = pd.DataFrame(data)
			except:
				raise ValueError("Data could not be initialized")

	def __str__(self, symbol="$"):
		return "({} {}, {}{})".format(self.name, self.type, symbol, self.price)

	def __repr__(self):
		return self.__str__()

	def next_day(self):
		self.date = self.date + datetime.timedelta(days=1) if self.date else None

	def valuate(self, column="Open"):
		if self.type == "cash":
			pass
		elif self.date and not self.data.empty:
			data = self.data.loc[self.data["Date"] == self.date.isoformat()]
			if not data.empty:
				self.price = float(data.iloc[0][column])
			else:
				print("No price data for {} on {}".format(self, self.date.isoformat()))
		else:
			error = "Missing data for valuation of {}".format(self) if self.date else "Missing date for valuation"
			print(error)

class Stock(Asset):
	def __init__(self, name, data, date):

		# Attribute initialization
		super(Stock, self).__init__("stock", name, None, date, data, True, ["stock", "cash"])

		#Valuate the position if we can
		if self.data.empty:
			raise ValueError("Incompatible dataset")

		# Trying to find a viable starting price
		last_date = datetime.date.fromisoformat(self.data.iloc[-1]["Date"])
		while not self.price:

			# Terminate search for price data if we've reached the end
			if self.date > last_date:
				raise ValueError("No price data available after {}, use an earlier date".format(date.isoformat()))

			# Try to valuate the position at the current date, move to the next day
			self.valuate()
			self.next_day()

		# Reset date to date given
		self.date = datetime.date.fromisoformat(date) if type(date) == str else date

class Option(Asset):
	def __init__(self, underlying, strike, expiry):

		# Attribute initialization
		super(Option, self).__init__(
			asset_type = "option", 
			name = underlying.name, 
			price = None, 
			date = underlying.date, 
			data = None,
			fractional = False,
			viable_collateral = underlying.viable_collateral)
		self.underlying = underlying
		self.strike = float(strike)
		self.expiry = None

		# Determining expiry
		if type(expiry) == str:
			self.expiry = datetime.date.fromisoformat(expiry)
		elif type(expiry) == int:
			self.expiry = self.date + datetime.timedelta(days=expiry)
		elif type(expiry) == datetime.timedelta:
			self.expiry = self.date + expiry
		elif type(expiry) == datetime.date:
			self.expiry = expiry
		else:
			raise ValueError("Invalid expiry for option initialization")

		# Determining option price
		self.price = self.underlying.price

class Call(Option):
	def exercise(self):
		pass

	def expire(self):
		pass



class Position:
	"""
	Class representing a financial position in an asset

	Arguments:
		asset {Asset}:
			The asset underlying the position

		volume {Int / Float}:
			A numeric value indicating the volume of the position

		date {datetime.date / Str}:
			A datetime.date object or isoformatted date string ("YYYY-MM-DD")
			which informs the position of how to update its prices

		short {Bool}:
			A boolean value indicating whether the position is long or short
			True when the position is short, False when long

		collateral {Position}:
			A position that serves to act as collateral for a short position.
			Only necessary for short positions
	"""
	def __init__(self, asset, volume, date=None, short=False, collateral=None):

		# Attribute initialization
		self.asset = asset
		self.volume = volume
		self.basis = asset.price
		self.value = self.basis * self.volume
		self.collateral = collateral
		self.short = short
		self.date = date
		self.history = {}

		# Ensure that shorts have adequate collateral
		if self.short: 
			if not self.collateral:
				raise ValueError("Short position must have collateral")
			elif self.collateral.short:
				raise ValueError("Collateral for short positions must long")
			elif self.value > self.collateral.value:
				raise ValueError("Insufficient collateral for this short position")

		# Collateral is unnecessary for long positions
		elif collateral:
			raise ValueError("No collateral required for long positions")

		self.update_history()


	def __add__(self, other):

		# Handle numeric additions to a position
		if type(other) in [int, float]:
			if self.collateral:
				raise ValueError("Positions with collateral can not be modified with numeric types")
			self.volume += other
			self.value += other * self.basis
			self.update_history()
			return self

		# Error checking, ensure that the positions are compatible for addition
		elif self.asset.type != other.asset.type \
		or self.asset.name != other.asset.name \
		or self.asset.price != other.asset.price:
			raise ValueError("Can not add positions with different assets")
		elif self.short != other.short:
			raise ValueError("Can not add short and long positions")
		elif self.collateral:
				# Recursive error checking if collateral exists
				self.collateral += other.collateral

		self.value += other.value
		self.volume += other.volume
		self.basis = self.value / self.volume

		self.update_history()

		return self

	def __sub__(self, other):

		# Handle numeric subractions to position:
		if type(other) in [int, float]:
			if self.collateral:
				raise ValueError("Positions with collateral can not be modified with numeric types")
			elif other >= self.volume:
				raise ValueError("Position is only has {} units, can not subtract {} units").format(
					self.volume,
					other)
			self.volume -= other
			self.value -= other * self.basis
			self.update_history()
			return self

		# Error checking, ensure that the positions are compatible for subtraction
		elif self.asset.type != other.asset.type \
		or self.asset.name != other.asset.name \
		or self.asset.price != other.asset.price:
			raise ValueError("Can not subtract positions with different assets")
		elif self.short != other.short:
			raise ValueError("Can not subtract dissimilar positions (both long or both short)")
		elif self.collateral:
				# Recursive error checking if collateral exists
				self.collateral -= other.collateral

		self.value -= other.value
		self.volume -= other.volume

		self.update_history()

		return self

	def __str__(self, symbol="$"):
		return "({} x {} @ {}{})".format(self.asset.__str__(symbol), self.volume, symbol, round(self.basis, 2))

	def __repr__(self):
		return self.__str__()

	def valuate(self):
		"""
		Updates the value of the position based on the value
		of the underlying asset
		"""
		# The new valuation
		valuation = self.asset.price * self.volume

		# If the position is shorted, we need to ensure that the
		# value of the collateral matches
		if self.short:
			self.collateral.valuate()
			if self.collateral.value != self.value:
				raise ValueError("\n{}\nInsufficient collateral for this position".format(self))

		self.value = valuation
		self.update_history()

	def update_history(self):
		"""
		Updates the position's history based for the current
		date
		"""
		if self.date:
			new_self = deepcopy(self.value)
			self.history[self.date.isoformat()] = new_self

class Cash(Position):
	"""
	The cash class used within portfolios as the basis of valuation
	for all other assets. Configurable to different currencies.

	Arguments:
		quantity {Int / Float}:
			A numeric value indicating the quantity of cash represented
			by this position

		currency_code {Str}:
			A string of length 3 that represents the type of currency in
			the cash position. For more information on this topic, see
			the following: https://en.wikipedia.org/wiki/Currency_symbol

	"""
	def __init__(self, quantity, currency_code="USD"):
		"""
		For cash the price is always one, because it acts as the
		basis unit of value for all other positions in the portfolio
		"""

		# Creating the cash asset for the position
		cash_asset = Asset("cash", currency_code, 1)

		# Attribute inititalization
		super(Cash, self).__init__(cash_asset, quantity)
		self.symbol = \
			{
			"EUR": "€",
			"USD": "$",
			"JPY": "¥",
			"GBP": "£"
			}[currency_code]

		self.update_history()


	def __str__(self, _):
		return "({}{})".format(self.symbol, self.volume)


class Portfolio:
	"""
	Class representing a portfolio which carries a set of financial
	positions, and a history of those positions across time. Portfolios
	can enter and exit both long and short positions in any asset considered
	valid by the portfolio.

	Arguments:
		initial {float / int}: 
			A number value representing the initial capital in the 
			portfolio, or the principal investment

		start_date {datetime.date / Str}: 
			A datetime.date object which represents the starting 
			point of the portfolio. Can also be passed in as a date 
			isoformat string: "YYYY-MM-DD"

		basis {Str}:
			A string indicating the type of asset used as a basis
			for the valuation of the portfolio, and all of the assets
			contained within. default is $USD

		risk_free_rate {float}: 
			A float value indicating the annual growth of capital 
			that can be achieved without assuming any risk. Default 
			value is the average annual index return (8%)

		margin {float}: 
			A float value that acts as a multiplier of the total 
			account value to determine how much margin the account
			is elligible to be loaned at a given moment

		valid_assets {List(Str)}:
			A list of strings indicating the type of assets that this
			portfolio is capable of purchasing

	"""
	def __init__(self, initial, start_date, basis="USD", risk_free_rate=0.08, margin=None, valid_assets=None):

		# Accept string inputs for the start date
		start_date = datetime.date.fromisoformat(start_date) if type(start_date) == str else start_date

		# Initializing portfolio positions
		self.positions = {"cash": [Cash(initial, basis)]}
		self.date = start_date

		# Initializing history
		self.history = {}
		self.update_history()

		# Initializing other attributes
		self.basis = basis
		self.basis_symbol = self.positions["cash"][0].symbol
		self.risk_free_rate = risk_free_rate
		self.maintenance_requirement = margin
		self.valid_assets = [] if valid_assets == None else valid_assets

	def __str__(self):

		# Accumulate string representing all portfollio positions
		result = ""
		for asset_type, positions in self.positions.items():
			pos_string = ""
			for position in positions:
				pos_string += "{}, ".format(position.__str__(self.basis_symbol))
			result += "{}: {}\n".format(asset_type, pos_string[:-2])

		# If result is empty, declare the portfolio empty
		if result:
			return result[:-1]
		else:
			return "Empty portfolio"

	def __repr__(self):
		return self.__str__()

	def reset(self):
		"""
		Resets the portfolio to its initial conditions.
		"""
		start, initial = list(self.history.items())[0]
		self.positions = initial
		self.history = {start: initial}

	def list_positions(self, positions=None):
		"""
		Returns a list of all the portfolio's positions
		"""
		positions = positions if positions else self.positions

		result = []
		for asset_type in positions:
			for position in positions[asset_type]:
				result.append(position)

		return result

	def assets(self, positions=None):
		"""
		Returns a list of all assets owned in the portfolio
		"""
		positions = self.list_positions(positions) if positions else self.list_positions()

		assets = list(set([position.asset for position in positions]))

		return assets

	def update_history(self):
		"""
		Used to update the history of the portfolio on
		the portfolio's current date.
		"""
		new_positions = deepcopy(self.positions)
		for position in self.list_positions(new_positions):
			position.asset.data = None
			position.history = None
		self.history[self.date.isoformat()] = new_positions

	def synchronise(self):
		"""
		Synchronises the date of all positions and all assets contained
		within the portfolio to be the same date as the portfolio
		"""
		for position in self.list_positions():
			position.date = self.date
		for asset in self.assets():
			asset.date = self.date

	def next_day(self):
		"""
		Used to move the portfolio and all of its positions to the next
		day
		"""
		self.date += datetime.timedelta(days=1)
		self.synchronise()
		self.valuate()
		self.update_history()

	def invest(self, amount):
		"""
		Used to increase principal investment into the
		portfolio by directly depositing cash.

		Arguments:
			amount {int / float / Cash}: Either a numeric type or
			a Cash object used to determine the amount of cash
			to invest in the portfolio
		"""

		# Directly deposit cash into the portfolio
		self.positions["cash"][0] += amount

		self.update_history()

	def withdraw(self, amount):
		"""
		Used to withdraw cash from the portfolio. Returns the
		amount withdrawn.

		Arguments:
			amount {int / float / Cash}: Either a numeric type or
			a Cash object used to determine the amount of cash
			to withdraw from the portfolio

		Returns:
			withdrawal {Cash}: A Cash object containing the amount
			of cash withdrawn from the account.
		"""

		# Withdraw cash from the portfolio
		self.positions["cash"][0] -= amount
		self.update_history()

		return Cash(amount)

	def check_validity(self, asset):
		"""
		Given an asset, check the asset's validity within this
		portfolio

		Arguments:
			asset {Asset}: 
				an asset object whose validity is being checked

		Returns:
			validity {Bool}:
				A boolean value indicating the validity of the
				asset relative to this portfolio. All assets are
				valid if the portfolio contains no valid asset
				list
		"""
		# Checking the list for validity
		if self.valid_assets:
			return True if asset.type in self.valid_assets else False

		# All assets are valid if no list is present
		return True

	def find_position(self, asset, short=False):
		"""
		Given an asset, finds the current position in said
		asset. If no current position exists, returns None

		Arguments:
			asset {Asset}:
				an asset object which is used to search for
				a position

			short (Bool):
				indicates whether the position being searched
				for is long or short. Will only return positions
				that match this value.

		Returns:
			current_position (Position):
				returns the current position in the asset,
				if any exists. Returns None otherwise. 

		"""

		# Check if we have positions in any assets of this type
		if asset.type in list(self.positions.keys()):

			# Find the position whose asset matches the input asset, if any
			current_position = next(
				(position for position in self.positions[asset.type] if position.asset == asset),
				None)

			# If we're trying to find a short, we need to make sure the position
			# being returned is also a short
			if short:
				return current_position if current_position.short else None
			else:
				return current_position
		
		else:
			return None

	def find_collateral(self, asset, volume):
		"""
		Given an asset and volume, finds suitable collateral if any exists

		Arguments:
			asset {Asset}:
				an asset object which is used to search for
				the collateral

			volume (Int / Float):
				a numeric value indicating the quantity of the asset
				for which collateral needs to be found

		Returns:
			collateral (Position):
				returns a position object which acts as sufficient
				collateral to cover the asset at the given volume

		"""

		long_position = self.find_position(asset)

	def buy(self, asset, volume):
		"""
		*** LONG POSITIONS ONLY ***
		Used to enter into a new long position

		Arguments:
			asset {Asset}:
				An asset object which will act as the underlying
				asset for the position entered into

			volume (Int / Float):
				A numeric value indicating the amount of the
				asset to purchase

		Returns:
			Modifies the portfolio's positions by purchasing the asset,
			entering the position.
		"""

		# The position to be bought
		position = Position(asset, volume, self.date)

		# Find the current position in said asset, if any
		current_position = self.find_position(asset)

		# Valid purchases only
		if not self.check_validity(asset):
			raise ValueError("Can not purchase this type of asset")

		# Must have a positive purchase volume
		elif volume <= 0:
			raise ValueError("Not a valid purchase volume")

		# nenenene volume must be integer unless otherwise allowed
		elif not asset.fractional and type(volume) != int:
			raise ValueError("Can not purchase fractional volumes of this asset")

		# Try to add to current positions, otherwise add new position
		else:

			# Decrement cash by the value of the position
			self.positions["cash"][0] -= position.value

		# Try to add positions, otherwise add the position to the
		# positions dictionary, creating a new key for that asset
		# class if none already exists
		try:
			current_position += position
		except:
			if asset.type in self.positions:
				self.positions[asset.type].append(position)
			else:
				self.positions[asset.type] = [position]

		self.update_history()

	def sell(self, asset, volume):
		"""
		*** LONG POSITIONS ONLY ***
		Used to exit or partially exit an existing long position.

		Arguments:
			asset {Asset}:
				The asset object being sold

			volume (Int / Float):
				A numeric value indicating the amount of the
				asset to sell

		Returns:
			Modifies the portfolio's positions by selling the asset,
			exiting the position
		"""
		# The position to be sold
		position = Position(asset, volume, self.date)

		# Find the current position in said asset, if any
		current_position = self.find_position(asset)

		# Must have a positive purchase volume
		if volume <= 0:
			raise ValueError("Not a valid sell volume")

		# Volume must be integer unless otherwise allowed
		elif not asset.fractional and type(volume) != int:
			raise ValueError("Can not sell fractional volumes of this asset")

		# We can only sell a position if it is currently owned
		elif current_position:

			# We can't sell more than we own
			if volume > current_position.volume:
				raise ValueError("Position too small to sell {}, only owns {}").format(
					volume,
					current_position.volume)

			# 
			try:
				current_position -= position
				self.positions["cash"][0] += position.value
			except:
				raise ValueError("Error decrementing current position: {} can not be subtracted from {}").format(
					position,
					current_position)

		# The position attempting to be sold is not owned
		else:
			print("Unable to find compatible asset to sell")

		self.update_history()

	def short(self, asset, volume):
		"""
		*** SHORT POSITIONS ONLY ***
		Used to enter into a new short position. Sells an asset that
		the portfolio doesn't own, requiring collateral. 

		Arguments:
			asset {Asset}:
				An asset object which will act as the underlying
				asset for the position entered into

			volume (Int / Float):
				A numeric value indicating the amount of the
				asset to purchase

		Returns:
			Modifies the portfolio's positions by shorting the asset,
			entering the position
		"""
		raise NotImplementedError

		# The position to be shorted
		position = Position(asset, volume, self.date, short=True)

		# Find the current position in said asset, if any
		current_position = self.find_position(asset, short=True)

		# Valid purchases only
		if not self.check_validity(asset):
			raise ValueError("Can not purchase this type of asset")

		# Must have a positive purchase volume
		elif volume <= 0:
			raise ValueError("Not a valid purchase volume")

		# nenenene volume must be integer unless otherwise allowed
		elif not asset.fractional and type(volume) != int:
			raise ValueError("Can not purchase fractional volumes of this asset")

		# Try to add to current positions, otherwise add new position
		else:

			# Decrement cash by the value of the position
			self.positions["cash"][0] -= position.value

		# Try to add positions, otherwise add the position to the
		# positions dictionary, creating a new key for that asset
		# class if none already exists
		try:
			current_position += position
		except:
			if asset.type in self.positions:
				self.positions[asset.type].append(position)
			else:
				self.positions[asset.type] = [position]

		self.update_history()

	def cover(self, asset, volume):
		"""
		*** SHORT POSITIONS ONLY ***
		Used to exit or partially exist a short position by covering
		(repurchasing) the short-sold loaned assset.

		Arguments:
			asset {Asset}:
				An asset object which will act as the underlying
				asset for the position entered into

			volume (Int / Float):
				A numeric value indicating the amount of the
				asset to purchase

		Returns:
			Modifies the portfolio's positions by covering the short,
			exiting the position
		"""
		raise NotImplementedError

		# The short to be covered
		position = Position(asset, volume, self.date)

		# Find the current position in said asset, if any
		current_position = self.find_position(asset)

		# Must have a positive purchase volume
		if volume <= 0:
			raise ValueError("Not a valid sell volume")

		# Volume must be integer unless otherwise allowed
		elif not asset.fractional and type(volume) != int:
			raise ValueError("Can not sell fractional volumes of this asset")

		# We can only sell a position if it is currently owned
		elif current_position:

			# We can't sell more than we own
			if volume > current_position.volume:
				raise ValueError("Position too small to sell {}, only owns {}").format(
					volume,
					current_position.volume)

			# 
			try:
				current_position -= position
				self.positions["cash"][0] += position.value
			except:
				raise ValueError("Error decrementing current position: {} can not be subtracted from {}").format(
					position,
					current_position)

		# The position attempting to be sold is not owned
		else:
			print("Unable to find compatible asset to sell")

		self.update_history()


	def valuate(self):
		"""
		Automatically updates the values of all positions
		currently held.
		"""
		for asset in self.assets():
			asset.valuate()
		for position in self.list_positions():
			position.valuate()

		self.update_history()

	def _value(self, date):
		"""
		Value helper, internal use only
		"""

		# all_assets must be the assets from this date
		all_assets = self.list_positions(self.history[date] if date in self.history else [])

		value = sum([position.value for position in all_assets])
		return np.array([date, float(value)])

	def value(self, date=None):
		"""
		Returns total portfolio value at a specified date.
		If the date is not specified, returns an np.array containing
		total portfolio value at each date.
		"""
		if date:
			return self._value(date)[1]
		else:
			return np.array([self._value(date) for date in self.history])

	def print_positions(self, positions=None, date=None):
		"""
		prints a formatted version of current portfolio
		positions. Positions and dates can be given to
		alter output.
		"""

		# Can print alternate positions given at a different date
		print_date = date if date else "CURRENT"
		positions = positions if positions else self.positions

		# Print Body
		print("\n{} POSITIONS\n--------------------".format(print_date))
		print("total: {}{}".format(self.basis_symbol, self.value(date if date else self.date)))
		print_portfolio = Portfolio(0, datetime.date.today())
		print_portfolio.positions = positions
		print(print_portfolio)
		print("--------------------")


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
		y = value_history[:, 1].astype(float)
		plt.plot(x, y, label="Account Value")
		skip = int(len(x) / 5)
		xlocs = range(len(x))
		plt.xticks(xlocs[::skip], x[::skip])
		plt.xlabel("Date")
		plt.ylabel("Portfolio Value (USD)")
		plt.title("Portfolio Performance")
		plt.show()


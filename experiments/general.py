from algorithms import finance, algos, models
import matplotlib.pyplot as plt
import pandas as pd
import datetime

def main():

	# Initial portfolio conditions, runtime
	start_date = "2005-02-25"
	end_date = "2017-01-01"
	initial = 500000

	# Tracking SPY progress on same initial investment
	print("Begin SPY backtesting...")
	p = finance.Portfolio(initial, start_date)
	spy = finance.Stock("SPY", "data/spy.csv", start_date)
	purchase_volume = int(p.liquid().volume / spy.price)
	p.buy(spy, purchase_volume)
	while p.date < datetime.date.fromisoformat(end_date):
		p.next_day()
	p.sell(spy, purchase_volume)
	value_history = p.value(p.weekdays())
	x = value_history[:, 0]
	y = value_history[:, 1].astype(float)
	plt.plot(x, y, label="SPY")
	print("SPY backtesting complete.\n")

	# Default SPY covered calls algorithm performance
	name = "SPY CC Default"
	print("Begin {} backtesting...".format(name))
	p = finance.Portfolio(initial, start_date)
	algos.spy_covered_calls(p, end_date=end_date)
	value_history = p.value(p.weekdays())
	y = value_history[:, 1].astype(float)
	plt.plot(x, y, label=name)
	print("{} backtesting complete.\n".format(name))

	# Experimental CC strategies
	expiries = {
		0: 0,
		1: 0,
		2: 0,
		3: 0,
		4: 0
	}
	strike_adjustments = 3
	expiry_adjustments = 3

	for i in range(expiry_adjustments):
		for k in expiries.keys():
			expiries[k] = i
		for strike_adjustment in range(strike_adjustments):
			name = "SPY CC {} days, +{}".format(i, strike_adjustment)
			print("Begin {} backtesting...".format(name))
			p = finance.Portfolio(initial, start_date)
			algos.spy_covered_calls(
				p,
				start_date,
				end_date,
				expiries,
				strike_adjustment)
			value_history = p.value(p.weekdays())
			y = value_history[:, 1].astype(float)
			plt.plot(x, y, label=name)
			print("{} backtesting complete\n".format(name))


	# Building Graph
	plt.title("Spy vs Covered Calls")
	plt.legend()

	num_labels = 5
	skip = int(len(x) / num_labels)
	xlocs = range(len(x))
	if skip > 0:
		plt.xticks(xlocs[::skip], x[::skip])

	plt.xlabel("Date")
	plt.ylabel("Portfolio Value")

	plt.show()

if __name__ == "__main__":
	main()
from algorithms import finance, algos, models
import matplotlib.pyplot as plt
import pandas as pd
import datetime

def main():

	# Initial portfolio conditions, runtime
	start_date = "2005-11-10"
	end_date = "2006-12-10"
	initial = 50000

	# Spy covered calls algorithm performance
	print("Begin backtesting...")
	p = finance.Portfolio(initial, start_date, verbose=True)
	algos.spy_covered_calls(p, end_date=end_date)
	p.print_history(p.weekdays())
	value_history = p.value(p.weekdays())
	print("Algorithm 1 backtesting complete.")

	# Same amount invested in spy over the same duration
	print("Begin backtesting...")
	p2 = finance.Portfolio(initial, start_date)
	spy = finance.Stock("SPY", "data/spy.csv", start_date)
	purchase_volume = int(p2.liquid().volume / spy.price)
	p2.buy(spy, purchase_volume)
	while p2.date < datetime.date.fromisoformat(end_date):
		p2.next_day()
	p2.sell(spy, purchase_volume)
	value_history2 = p2.value(p.weekdays())
	print("Algorithm 2 backtesting complete.")

	# Building out graphs
	x = value_history[:, 0]
	y1 = value_history[:, 1].astype(float)
	y2 = value_history2[:, 1].astype(float)

	plt.title("Spy vs Covered Calls")
	plt.plot(x, y1, label="Covered Calls")
	plt.plot(x, y2, label="SPY")
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
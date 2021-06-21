from algorithms import finance, algos, models
import datetime

def main():

	# Testing
	myp = finance.Portfolio(100000, "2000-01-01", basis="JPY")
	spy = finance.Asset("stock", "spy", 300)
	print(myp.positions["cash"][0].symbol)
	myp.next_day()
	myp.buy(spy, 100)
	myp.print_positions()

	myp.reset()
	myp.print_positions()

	"""
	# Running the algorithm
	covered_calls = finance.Portfolio(40000, "2005-02-24")
	algos.spy_covered_calls(covered_calls)

	# Performance Diagnostics
	covered_calls.print_history()
	covered_calls.show_history()
	"""

if __name__ == "__main__":
	main()
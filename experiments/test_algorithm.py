from algorithms import finance, algos, models
import pandas as pd
import datetime

def main():

	# Testing
	start_date = "2005-02-25"
	myp = finance.Portfolio(100000, start_date, basis="JPY")
	spy = finance.Stock("SPY", "data/spy.csv", start_date)
	spy_call = finance.Call(spy, 110, 30)
	myp.buy(spy, 400)
	myp.buy(spy_call, 400)

	for x in range(1000):
		myp.next_day()


	myp.sell(spy, 399)
	myp.print_history()
	myp.show_history()


if __name__ == "__main__":
	main()
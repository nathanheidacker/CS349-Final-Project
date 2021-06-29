from algorithms import finance, algos, models
import pandas as pd
import datetime

def main():

	p = finance.Portfolio(50000, "2005-02-25", verbose=True)
	algos.spy_covered_calls(p, end_date=1000)
	p.print_history()
	p.show_history()
	

if __name__ == "__main__":
	main()
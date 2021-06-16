from algorithms import finance, algos, models


def main():
	covered_calls = finance.Portfolio(40000, "2005-02-24")
	algos.spy_covered_calls(covered_calls)
	covered_calls.print_history()
	covered_calls.show_history()

if __name__ == "__main__":
	main()
from algorithms import finance, algos

def main():
	import cProfile
	import pstats

	p = finance.Portfolio(100000, "2005-02-25")

	with cProfile.Profile() as pr:
		algos.spy_covered_calls(p, end_date="2017-11-10")

	stats = pstats.Stats(pr)
	stats.sort_stats(pstats.SortKey.TIME)
	stats.print_stats()

if __name__ == '__main__':
	main()

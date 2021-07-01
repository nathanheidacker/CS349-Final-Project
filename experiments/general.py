from algorithms import finance, algos, models
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import copy

positions = finance.Positions()
spy = finance.Stock("SPY", "data/spy.csv", "2005-02-25")
positions["cash"] = [finance.Cash(10000)]
positions["stock"] = [finance.Position(spy, 100)]

positions_copy = copy.deepcopy(positions)

print(positions)
print(positions_copy, "\n")

positions_copy["cash"][0] += 10000

print(positions)
print(positions_copy, "\n")

positions["cash"][0] += 50000

print(positions)
print(positions_copy, "\n")
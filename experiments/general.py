from algorithms import finance, algos, models
from data import datatools
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import copy

print(datatools.format_asset_dates("1999/25/02"))

print(finance.Stock("SPY"))
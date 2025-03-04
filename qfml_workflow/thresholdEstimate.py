# etf trick

# use last n number of bars too determine threshold levl, maybe use
# price tiems shares times the perdiod thqe bbars spanned over, \
# so that it can be weighted as a function of the time?

# have price factor (x percent of total dv over time)
# and have a time/period factor (bars should encompass n periods on  avg)
# give average bars sampled per day as a metric when making dbars
'''
tp = 0.25 days
10000 dv
factor = 0.01
1 day
'''

# dv / td * factor
# or use weighted ma of dv over n trades, might be bad because relies on the 
# factor used
from qfml_workflow import dataStructures
import numpy as np
import pandas as pd

infp = 'D:/prado research data/trades_long_condensed/'
outfp = 'E:/Prado Research/qfml_workflow/data/'

tickers = ["EWA", "EWK", "EWO", "EWC", "EWQ", "EWG", "EWH", "EWI", "EWJ", "EWM", "EWW", 
                    "EWN", "EWS", "EWP", "EWD", "EWL", "EWY", "EZU", "EWU", "EWZ", "EWT",]
#tickers = ['EWA']
for ticker in tickers:
    dataStructures.dollar_bars(infp+ticker+'_trades_condensed.csv', 15000000, chunksize=1000000, outfp=outfp+ticker+'_hour.csv')
import pandas as pd
from qfml_workflow import dataStructures

filepath = 'D:/prado research data/trades_long_condensed/SPY_trades_condensed.csv'

sample_thresh = 7E9
chunksize = 2000000

outfp = 'E:/Prado Research/qfml_workflow/data/SPY_long_high_window/SPY_full_dbars.csv'

dataStructures.dollar_bars(filepath, sample_thresh=sample_thresh, 
                           chunksize=chunksize, outfp=outfp)
import numpy as np
import pandas as pd
import pyfolio as pf

def get_daily_returns(intraday_returns):
    #TODO: use a more robust method of returns
    cum_rets = ((intraday_returns + 1).cumprod())

    # Downsample to daily
    daily_rets = cum_rets.resample('B').last()

    # Forward fill, Percent Change, Drop NaN
    daily_rets = daily_rets.ffill().pct_change().dropna()
    
    return daily_rets

def getIntradayReturns(signal, close):
    return signal * np.log(close).diff()


def simplePerformanceReport(returns, daily=False):
    if daily == False:
        dailyRet = get_daily_returns(returns)
        pf.show_perf_stats(returns=dailyRet, factor_returns=None)
    else:
        pf.show_perf_stats(returns=returns, factor_returns=None)

def fullPerformanceTearSheet(returns, close, daily=False):
    benchmarkRet = np.log(close).diff()
    benchmarkRet = benchmarkRet.dropna()
    if daily == False:
        dailyRet = get_daily_returns(returns)
        dailyRet = dailyRet.dropna()
        pf.create_returns_tear_sheet(dailyRet, benchmark_rets = benchmarkRet)
    else:
        pf.create_returns_tear_sheet(returns, benchmark_rets = benchmarkRet )
    
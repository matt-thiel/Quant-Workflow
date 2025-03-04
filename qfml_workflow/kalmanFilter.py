import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import talib as ta
import yfinance as yf

def compute_spread(Y, gamma, mu, name= np.NaN):
    w1 = pd.DataFrame({'x1': 1, 'gamma': -gamma})
    w2 = pd.DataFrame({'x1': 1+gamma, 'gamma': 1+gamma})
    w_spread = w1/w2
    spread = (Y * w_spread).sum() - mu/(1+gamma)
    return spread

def generate_Z_score_EMA(spread, n=120):
    spread_mean = ta.EMA(spread, n)
    # backfill na
    spread_mean = spread_mean.bfill(axis=0)

    spread_demeaned = spread - spread_mean

    # variance
    spread_var = ta.EMA(spread_demeaned**2, n)
    # backfill na
    spread_var = spread_var.bfill(axis=0)

    Zscore = spread_demeaned / np.sqrt(spread_var)
    return Zscore

def generate_signal(Z_score, threshold_long, threshold_short):
    #signal = pd.Series([None for x in range(0, len(Z_score))], index = Z_score.index)
    signal = pd.Series(index = Z_score.index, dtype='float64')

    #initial position
    signal[0] = 0
    if Z_score[0] <= threshold_long[0]:
        signal[0] = 1
    elif (Z_score[0] >= threshold_short[0]):
        signal[0] = -1

    # Loop
    for t in range(1, len(Z_score)):
        if (signal[t-1] == 0): # no position
            if (Z_score[t] <= threshold_long[t]):
                signal[t] = 1
            elif (Z_score[t] >= threshold_short[t]):
                signal[t] = -1
            else:
                signal[t] = 0
        elif (signal[t-1] == 1): # already in long position
            if (Z_score[t] >= 0):
                signal[t] = 0
            else:
                signal[t] = signal[t-1]
        else: # already in short position
            if (Z_score[t] <= 0):
                signal[t] = 0
            else:
                signal[t] = signal[t-1]

    # return a series for shifting reasons
    return pd.Series(signal)

def estimate_mu_gamma_LS(Y):
    trainLen = round(0.3 * len(Y))
    rLP = Y[0:trainLen]
    t1, t2 = Y.columns.values
    f = t1 + "~" + t2

    model = smf.ols(formula=f, data=rLP).fit()
    
    mu, gamma = model.params

    return mu, gamma

def estimate_mu_gamma_kalman(Y):
    Tt = np.eye(2)
    Rt = np.eye(2)
    Qt = 1e-5*np.eye(2)  # state transition variance very small
    #trY = Y.copy()
    #trY.iloc[:, 0]  = 1

    #Zt = trY.to_numpy()
    Ht = np.matrix(1e-3)  # observation variance
    # the prior in the code: P1cov = kappa*P1Inf + P1, kappa = 1e7
    init_mu, init_gamma = estimate_mu_gamma_LS(Y)
    print("init mu and gamma")
    print(init_mu)
    print(init_gamma)
    a1 = np.matrix([init_mu, init_gamma])
    P1 = 1e-5*np.eye(2)  # variance of initial point
    P1inf = 0*np.eye(2)

    kf = sm.tsa.statespace.MLEModel(pd.DataFrame(Y.iloc[:,0]), k_states=2)

    #kf._state_names = ['x1', 'dx1/dt', 'x2', 'dx2/dt']
    kf._state_names = ['mean', 'gamma']
    #kf['design'] = np.c_[np.ones(len(prices[etfs[1]])), prices[etfs[1]]].T[np.newaxis, :, :]
    kf['design'] = np.c_[np.ones(len(Y.iloc[:, 1])), Y.iloc[:, 1]].T[np.newaxis, :, :]
    kf['obs_cov'] = Ht
    kf['transition'] = Tt
    kf['selection'] = Rt
    kf['state_cov'] = Qt

    # Edit: the timing convention for initialization
    # in Statsmodels differs from the the in the question
    # So we should not use kf.initialize_known(m0[:, 0], P0)
    # But instead, to fit the question's initialization
    # into Statsmodels' timing, we just need to use the
    # transition equation to move the initialization
    # forward, as follows:
    #kf.initialize_known(A @ m0[:, 0], A @ P0 @ A.T + Q)
    kf.initialize_known([init_mu, init_gamma], P1)
    #kf.initialize_known(Tt @ init_mu, Tt @ P1 @ Tt.T + Qt)
    # To performan Kalman filtering and smoothing, use:
    #res = kf.smooth(prices[etfs[0]].values)
    res = kf.smooth(Y.iloc[:, 0].values)

    # output vectors
    muF = res.filtered_state[0]
    gammaF = res.filtered_state[1]

    # smooth values
    muS = pd.Series(ta.SMA(muF, 30))
    gammaS = pd.Series(ta.SMA(gammaF, 30))

    # backfill values
    muS = muS.bfill(axis=0)
    gammaS = gammaS.bfill(axis=0)

    return muS, gammaS

def pairs_trading(Y, gamma, mu, name=np.NaN, threshold= 0.7, plot= False, outdata=False, use_opt_threshold=False):
    w1 = pd.DataFrame({'x1': 1, 'gamma': -gamma})
    w2 = pd.DataFrame({'x1': 1+gamma, 'gamma': 1+gamma})
    w_spread = w1/w2

    #print(gamma)
    #print(mu)

    w_spread = w_spread.set_index(Y.index)
    Y.columns = ['x1', 'gamma']
    # same as compute spread
    spread = (Y * w_spread).sum(axis=1) - mu.values/(1+gamma.values)

    # thresholds should be a list of same value the same size as Z_score
    Z_score = generate_Z_score_EMA(spread)
    # Python workaround
    #threshold_long = Z_score
    #threshold_short = Z_score
    #threshold_short = threshold
    #threshold_long = -threshold

    if use_opt_threshold:
        s0 = np.linspace(0, max(Z_score), 50)

        f_bar = np.array([None]*50)
        for i in range(50):
            f_bar[i] = len(Z_score.values[Z_score.values > s0[i]]) / Z_score.shape[0]

        D = np.zeros((49, 50))
        for i in range(D.shape[0]):
            D[i, i] = 1
            D[i, i+1] = -1

        l = 1.0

        f_star = np.linalg.inv(np.eye(50) + l * D.T@D) @ f_bar.reshape(-1, 1)
        s_star = [f_star[i]*s0[i] for i in range(50)]

        threshold = s0[s_star.index(max(s_star))]
        print(f"The optimal threshold is {threshold}")

    threshold_long = Z_score.copy()
    threshold_short = Z_score.copy()
    threshold_short[:] = threshold
    threshold_long[:] = -threshold
 
    signal = generate_signal(Z_score, threshold_long, threshold_short)

    # multiply both columns by the signal
    # monkey workaround
    #w_portf = w_spread * signal.shift(1)
    signal = signal.shift(1)
    
    #w_portf = w_spread.apply(lambda x: x * signal)
    w_portf = w_spread.mul(signal, axis=0)

    return signal, w_spread, spread

    X = Y.diff()
    # needs to be checked
    portf_return = (X * w_portf).sum(axis=1)
    portf_return = portf_return.fillna(0)
    # portf_return.columns = name

    #portf_return = portf_return.iloc[200:]

    if plot:
        np.cumprod(1 + portf_return).plot()
        #plt.plot(Z_score)
    #if outdata:
        #pd.to_csv(signal)
        #return np.cumprod(1 + portf_return)

    #return portf_return
'''       
tst = ['SPY', 'QQQ']
#dl1 = yf.download(tst[0], start = '2000-08-01', end='2004-1-1', auto_adjust=True).Close
#dl2 = yf.download(tst[1], start = '2000-08-01', end='2004-1-1', auto_adjust=True).Close
dl1 = yf.download(tst[0], start = '2010-08-01', end='2013-1-1', auto_adjust=True).Close
dl2 = yf.download(tst[1], start = '2010-08-01', end='2013-1-1', auto_adjust=True).Close

prices = pd.DataFrame({tst[0] : dl1, tst[1]: dl2})

prices = np.log(prices)
prices = prices.dropna()
Y = prices

mu, gamma = estimate_mu_gamma_kalman(Y)
gammaR = pd.read_csv("C:/Users/mathe/Downloads/gammaR.csv", index_col='Index')
#gammaR.index = pd.to_datetime(gammaR.index)
gammaR = gammaR['gamma-Kalman']
muR = pd.read_csv("C:/Users/mathe/Downloads/muR.csv", index_col='Index')
#muR.index = pd.to_datetime(muR.index)
muR = muR['mu-Kalman']
YR =  pd.read_csv("C:/Users/mathe/Downloads/Y_R.csv", index_col='Index')
YR = YR.loc[gammaR.index]
#pairs_trading(YR, gammaR, muR, plot=True, outdata=True, use_opt_threshold=True)
# for plotting purposes

pairs_trading(Y, gamma, mu, plot=True)
'''

etfs = ['EWA', 'EWC']
dfewa = pd.read_csv('E:\Prado Research\qfml_workflow\data\EWA.csv').drop(['num_ticks_sampled','tick_num'], axis=1)
dfewc = pd.read_csv('E:\Prado Research\qfml_workflow\data\EWC.csv').drop(['num_ticks_sampled','tick_num'], axis=1)

df1 = dfewa[['close', 'timestamp']]
df2 = dfewc[['close', 'timestamp']]

# uncomment below for daily
#dfewa['timestamp'] = pd.to_datetime(dfewa['timestamp'])
#dfewc['timestamp'] = pd.to_datetime(dfewc['timestamp'])
#df1['timestamp'] = pd.to_datetime(df1['timestamp'])
#df2['timestamp'] = pd.to_datetime(df2['timestamp'])

aggDict = {'open':'first',
           'high':'max',
           'low': 'min',
           'close': 'last',
           'volume': 'sum'}

#dfewa = dfewa.set_index('timestamp', drop=True).resample('B').agg(aggDict)
#dfewc = dfewc.set_index('timestamp', drop=True).resample('B').agg(aggDict)




dfewa['timestamp'] = pd.to_datetime(dfewa['timestamp']).round('60min')
dfewc['timestamp'] = pd.to_datetime(dfewc['timestamp']).round('60min')

dfewa = dfewa[~dfewa.index.duplicated(keep='last')]
dfewc = dfewc[~dfewc.index.duplicated(keep='last')]


dfewa.to_csv('E:\Prado Research\qfml_workflow\data\kalman_updated\EWA_hour_resample.csv')
dfewc.to_csv('E:\Prado Research\qfml_workflow\data\kalman_updated\EWC_hour_resample.csv')

df1['timestamp'] = pd.to_datetime(df1['timestamp']).round('60min')
df2['timestamp'] = pd.to_datetime(df2['timestamp']).round('60min')

df1 = df1.set_index('timestamp', drop=True)
df2 = df2.set_index('timestamp', drop=True)
df1 = df1[~df1.index.duplicated(keep='last')]
df2 = df2[~df2.index.duplicated(keep='last')]
#df1 = df1.set_index('timestamp', drop=True).resample('B').last()
#df2 = df2.set_index('timestamp', drop=True).resample('B').last()

y = pd.concat([df1, df2], axis=1)
y.columns = ['EWA', 'EWC']

print(y.head())
y = y.dropna()

mu, gamma = estimate_mu_gamma_kalman(y)
signal, weights, spread = pairs_trading(y, gamma, mu, plot=True)

kalFrame = pd.DataFrame({'signal':signal, 'EWAweights': weights['x1'], 'EWCweights': weights['gamma']}, index=weights.index)
kalFrame.to_csv('E:\Prado Research\qfml_workflow\data\kalman_updated\kalmanOutputHour.csv')
print(len(kalFrame))
print(len(y.dropna()))
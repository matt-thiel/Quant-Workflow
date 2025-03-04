import pandas as pd
from scipy.stats import norm, moment
from qfml_workflow.multiprocess import mpPandasObj
import numpy as np

def getSignal(events,stepSize,prob,pred,numClasses,numThreads,**kargs):
    # get signals from predictions
    if prob.shape[0]==0:return pd.Series()
    #1) generate signals from multinomial classification (one-vs-rest, OvR)
    signal0=(prob-1./numClasses)/(prob*(1.-prob))**.5 # t-value of OvR
    signal0=pred*(2*norm.cdf(signal0)-1) # signal=side*size
    if 'side' in events:signal0*=events.loc[signal0.index,'side'] # meta-labeling
    #2) compute average signal among those concurrently open
    df0=signal0.to_frame('signal').join(events[['t1']],how='left')
    df0=avgActiveSignals(df0,numThreads)
    signal1=discreteSignal(signal0=df0,stepSize=stepSize)
    return signal1

def getSignalNoSide(events,stepSize,prob,pred,numClasses,numThreads,**kargs):
    # get signals from predictions
    if prob.shape[0]==0:return pd.Series()
    #1) generate signals from multinomial classification (one-vs-rest, OvR)
    signal0=(prob-1./numClasses)/(prob*(1.-prob))**.5 # t-value of OvR
    signal0=pred*(2*norm.cdf(signal0)-1) # signal=side*size
    #if 'side' in events:signal0*=events.loc[signal0.index,'side'] # meta-labeling
    #2) compute average signal among those concurrently open
    df0=signal0.to_frame('signal').join(events[['t1']],how='left')
    df0=avgActiveSignals(df0,numThreads)
    signal1=discreteSignal(signal0=df0,stepSize=stepSize)
    return signal1

def avgActiveSignals(signals,numThreads):
    # compute the average signal among those active
    #1) time points where signals change (either one starts or one ends)
    tPnts=set(signals['t1'].dropna().values)
    tPnts=tPnts.union(signals.index.values)
    tPnts=list(tPnts);tPnts.sort()
    out=mpPandasObj(mpAvgActiveSignals,('molecule',tPnts),numThreads,signals=signals)
    return out
#———————————————————————————————————————
def mpAvgActiveSignals(signals,molecule):
    '''
    At time loc, average signal among those still active.
    Signal is active if:
    a) issued before or at loc AND
    b) loc before signal’s endtime, or endtime is still unknown (NaT).
    '''
    out=pd.Series()
    for loc in molecule:
        df0=(signals.index.values<=loc)&((loc<signals['t1'])|pd.isnull(signals['t1']))
        act=signals[df0].index
        if len(act)>0:out[loc]=signals.loc[act,'signal'].mean()
        else:out[loc]=0 # no signals active at this time
    return out

def discreteSignal(signal0,stepSize):
    # discretize signal
    signal1=(signal0/stepSize).round()*stepSize # discretize
    signal1[signal1>1]=1 # cap
    signal1[signal1<-1]=-1 # floor
    return signal1

def betSize(w,x):
    return x*(w+x**2)**-.5
#———————————————————————————————————————
def getTPos(w,f,mP,maxPos):
    return int(betSize(w,f-mP)*maxPos)
#———————————————————————————————————————
def invPrice(f,w,m):
    return f-m*(w/(1-m**2))**.5
#———————————————————————————————————————
def limitPrice(tPos,pos,f,w,maxPos):
    sgn=(1 if tPos>=pos else -1)
    lP=0
    for j in range(abs(pos+sgn),abs(tPos+1)):
        lP+=invPrice(f,w,j/float(maxPos))
    lP/=tPos-pos
    return lP
#———————————————————————————————————————
def getW(x,m):
# 0<alpha<1
    return x**2*(m**-2-1)
#———————————————————————————————————————


# Snippet 10.4, modified to use a power function for the Bet Size
# ===============================================================
# pos    : current position
# tPos   : target position
# w      : coefficient for regulating width of the bet size function (sigmoid, power)
# f      : forecast price
# mP     : market price
# x      : divergence, f - mP
# maxPos : maximum absolute position size
# ===============================================================

def betSize_power(w, x):
    # returns the bet size given the price divergence
    sgn = np.sign(x)
    return sgn * abs(x)**w

def getTPos_power(w, f, mP, maxPos):
    # returns the target position size associated with the given forecast price
    return int( betSize_power(w, f-mP)*maxPos )

def invPrice_power(f, w, m):
    # inverse function of bet size with respect to the market price
    sgn = np.sign(m)
    return f - sgn*abs(m)**(1/w)

def limitPrice_power(tPos, pos, f, w, maxPos):
    # returns the limit price given forecast price
    sgn = np.sign(tPos-pos)
    lP = 0
    for j in range(abs(pos+sgn), abs(tPos+1)):
        lP += invPrice_power(f, w, j/float(maxPos))
    lP = lP / (tPos-pos)
    return lP

def getW_power(x, m):
    # inverse function of the bet size with respect to the 'w' coefficient
    return np.log(m/np.sign(x)) / np.log(abs(x))
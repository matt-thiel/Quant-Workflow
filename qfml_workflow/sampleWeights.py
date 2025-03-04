import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotnine as pn
from qfml_workflow.multiprocess import mpPandasObj, processJobs_, processJobs

def getWeights(data, events, cpus=1):
    """
    Get sample weights for triple barrier events

    :param DataFrame data: The person sending the message
    :param Series events: The recipient of the message
    :param cpus: Number of cpus to use for multiprocessing
    :return: DataFrame with sample weights
    :rtype: DataFrame
    #:raises ValueError: if the message_body exceeds 160 characters
    #:raises TypeError: if the message_body is not a basestring
    """
    numCoEvents = getCoEvents(data, events, cpus=cpus)

    out=pd.DataFrame()
    out['tW'] = mpPandasObj(mpSampleTW,('molecule',events.index),
                                cpus,t1=events['t1'],numCoEvents=numCoEvents)
    ## example ##
    out['w']= mpPandasObj(mpSampleW,('molecule',events.index),cpus,
                            t1=events['t1'],numCoEvents=numCoEvents,close=data['close'])
    out['w']*= out.shape[0]/out['w'].sum()

    return out

def mpNumCoEvents(closeIdx,t1,molecule):
    '''
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed
    
    Any event that starts before t1[modelcule].max() impacts the count.
    '''
    #1) find events that span the period [molecule[0],molecule[-1]]
    t1=t1.fillna(closeIdx[-1]) # unclosed events still must impact other weights
    t1=t1[t1>=molecule[0]] # events that end at or after molecule[0]
    t1=t1.loc[:t1[molecule].max()] # events that start at or before t1[molecule].max()
    #2) count events spanning a bar
    iloc=closeIdx.searchsorted(np.array([t1.index[0],t1.max()]))
    count=pd.Series(0,index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn,tOut in t1.iteritems():count.loc[tIn:tOut]+=1.
    return count.loc[molecule[0]:t1[molecule].max()]

def mpSampleTW(t1,numCoEvents,molecule):
    # Derive avg. uniqueness over the events lifespan
    wght=pd.Series(index=molecule)
    for tIn,tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn]=(1./numCoEvents.loc[tIn:tOut]).mean()
    return wght

# sample weight by return attribution. neutral cases may need to be dropped
def mpSampleW(t1,numCoEvents,close,molecule):
    # Derive sample weight by return attribution
    ret=np.log(close).diff() # log-returns, so that they are additive
    wght=pd.Series(index=molecule)
    for tIn,tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn]=(ret.loc[tIn:tOut]/numCoEvents.loc[tIn:tOut]).sum()
    return wght.abs()

# ------ Sequential Bootstrapping
def getIndMatrix(barIx,t1):
    # Get indicator matrix
    indM=pd.DataFrame(0,index=barIx,columns=range(t1.shape[0]))
    for i,(t0,t1) in enumerate(t1.iteritems()):indM.loc[t0:t1,i]=1.
    return indM

def getAvgUniqueness(indM):
    # Average uniqueness from indicator matrix
    c=indM.sum(axis=1) # concurrency
    u=indM.div(c,axis=0) # uniqueness
    avgU=u[u>0].mean() # average uniqueness
    return avgU

def seqBootstrap(indM,sLength=None):
    # Generate a sample via sequential bootstrap
    if sLength is None:sLength=indM.shape[1]
    phi=[]
    while len(phi)<sLength:
        avgU=pd.Series()
        for i in indM:
            indM_=indM[phi+[i]] # reduce indM
            avgU.loc[i]=getAvgUniqueness(indM_).iloc[-1]
        prob=avgU/avgU.sum() # draw prob
        phi+=[np.random.choice(indM.columns,p=prob)]
    return phi

#------- Extra Func
def getCoEvents(data, events, cpus=1):
    numCoEvents = mpPandasObj(mpNumCoEvents,('molecule',events.index),                         
                              cpus,closeIdx=data['close'].index,t1=events['t1'])
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
    numCoEvents = numCoEvents.reindex(data['close'].index).fillna(0)

    return numCoEvents


#--------------------- Time decay
def getTimeDecay(tW,clfLastW=1.):
    # apply piecewise-linear decay to observed uniqueness (tW)
    # newest observation gets weight=1, oldest observation gets weight=clfLastW
    clfW=tW.sort_index().cumsum()
    if clfLastW>=0: slope=(1.-clfLastW)/clfW.iloc[-1]
    else: slope=1./((clfLastW+1)*clfW.iloc[-1])
    const=1.-slope*clfW.iloc[-1]
    clfW=const+slope*clfW
    clfW[clfW<0]=0
    print(const,slope)
    return clfW

def getExTimeDecay(tW,clfLastW=1.,exponent=1):
    # apply exponential decay to observed uniqueness (tW)
    # newest observation gets weight=1, oldest observation gets weight=clfLastW
    clfW=tW.sort_index().cumsum()
    if clfLastW>=0: slope=((1.-clfLastW)/clfW.iloc[-1])**exponent
    else: slope=(1./((clfLastW+1)*clfW.iloc[-1]))**exponent
    const=1.-slope*clfW.iloc[-1]
    clfW=const+slope*clfW
    clfW[clfW<0]=0
    print(round(const,4), round(slope,4))
    return clfW

def plotDecay(uniquenessWeights, clfLastW, exponent):
    f,ax=plt.subplots(2,figsize=(10,7))
    fs = [1,.75,.5,0,-.25,-.5]
    ls = ['-','-.','--',':','--','-.']
    for lstW, l in zip(fs,ls):
        decayFactor = getExTimeDecay(uniquenessWeights['tW'].dropna(), 
                                    clfLastW=lstW,
                                    exponent=0.75) # experiment by changing exponent
        ((uniquenessWeights['w'].dropna()*decayFactor).reset_index(drop=True)
        .plot(ax=ax[0],alpha=0.5))
        s = (pd.Series(1,index=uniquenessWeights['w'].dropna().index)*decayFactor)
        s.plot(ax=ax[1], ls=l, label=str(lstW))
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))


#----------- Monte Carlo
def getRndT1(numObs,numBars,maxH):
    # random t1 Series
    t1=pd.Series()
    # xrange in python2
    for i in range(numObs):
        ix=np.random.randint(0,numBars)
        val=ix+np.random.randint(1,maxH)
        t1.loc[ix]=val
    return t1.sort_index()

def auxMC(numObs,numBars,maxH):
    # Parallelized auxiliary function
    t1=getRndT1(numObs,numBars,maxH)
    barIx=range(t1.max()+1)
    indM=getIndMatrix(barIx,t1)
    phi=np.random.choice(indM.columns,size=indM.shape[1])
    stdU=getAvgUniqueness(indM[phi]).mean()
    phi=seqBootstrap(indM)
    seqU=getAvgUniqueness(indM[phi]).mean()

    return {'stdU':stdU,'seqU':seqU}

#---------- Monte Carlo with multithreading
def mainMC(numObs=10,numBars=100,maxH=5,numIters=1E6,numThreads=24):
    # Monte Carlo experiments
    jobs=[]
    for i in range(int(numIters)):
        job={'func':auxMC,'numObs':numObs,'numBars':numBars,'maxH':maxH}
        jobs.append(job)
    if numThreads==1:out=processJobs_(jobs)
    else:out=processJobs(jobs,numThreads=numThreads)
    print( pd.DataFrame(out).describe())
    return
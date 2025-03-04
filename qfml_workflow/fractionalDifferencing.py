import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

def getWeights(d,size):
    # thres>0 drops insignificant weights
    w=[1.]
    for k in range(1,size):
        w_=-w[-1]/k*(d-k+1)
        w.append(w_)
    w=np.array(w[::-1]).reshape(-1,1)
    return w

def getWeights_FFD(d=0.1, thres=1e-5):
    w,k = [1.],1
    while True:
        w_=-w[-1]/k*(d-k+1)
        if abs(w_)<thres:break
        w.append(w_)
        k+=1
    return np.array(w[::-1]).reshape(-1,1)

def plotWeights(dRange,nPlots,size):
    w=pd.DataFrame()
    for d in np.linspace(dRange[0],dRange[1],nPlots):
        w_=getWeights(d,size=size)
        w_=pd.DataFrame(w_,index=range(w_.shape[0])[::-1],columns=[d])
        w=w.join(w_,how='outer')
    ax=w.plot()
    ax.legend(loc='upper left')
    plt.show()
    return

def fracDiff(series,d,thres=.01):
    '''
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    #1) Compute weights for the longest series
    w=getWeights(d,series.shape[0])
    #2) Determine initial calcs to be skipped based on weight-loss threshold
    w_=np.cumsum(abs(w))
    w_/=w_[-1]
    skip=w_[w_>thres].shape[0]
    #3) Apply weights to values
    df={}
    for name in series.columns:
        seriesF,df_=series[[name]].fillna(method='ffill').dropna(),pd.Series()
        for iloc in range(skip,seriesF.shape[0]):
            loc=seriesF.index[iloc]
            if not np.isfinite(series.loc[loc,name]):continue # exclude NAs
            df_[loc]=np.dot(w[-(iloc+1):,:].T,seriesF.loc[:loc])[0,0]
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df

# fixed width window fracdiff
def fracDiff_FFD(series,d,thres=1e-5):
    '''
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    #1) Compute weights for the longest series
    # should this be the regular getWeights?
    w=getWeights_FFD(d,thres)
    #w=getWeights(d,thres)
    width=len(w)-1
    #2) Apply weights to values
    df={}
    for name in series.columns:
        seriesF,df_=series[[name]].fillna(method='ffill').dropna(),pd.Series()
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1=seriesF.index[iloc1-width],seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):continue # exclude NAs
            df_[loc1]=np.dot(w.T,seriesF.loc[loc0:loc1])[0,0]
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df

def plotMinFFD(data, outpath, outname='ES1_Index_Method12'):
    from statsmodels.tsa.stattools import adfuller
    path,instName=outpath, outname
    out=pd.DataFrame(columns=['adfStat','pVal','lags','nObs','95% conf','corr'])
    #df0=pd.read_csv(path+instName+'.csv',index_col=0,parse_dates=True)
    df0 = data
    for d in np.linspace(0,1,11):
        df1=np.log(df0[['close']]).resample('1D').last() # downcast to daily obs
        df2=fracDiff_FFD(df1,d,thres=.01)
        corr=np.corrcoef(df1.loc[df2.index,'close'],df2['close'])[0,1]
        df2=adfuller(df2['close'],maxlag=1,regression='c',autolag=None)
        out.loc[d]=list(df2[:4])+[df2[4]['5%']]+[corr] # with critical value
    out.to_csv(path+instName+'_testMinFFD.csv')
    out[['adfStat','corr']].plot(secondary_y='adfStat')
    plt.axhline(out['95% conf'].mean(),linewidth=1,color='r',linestyle='dotted')
    plt.savefig(path+instName+'_testMinFFD.png')
    return

def getMinFFD(data):
    from statsmodels.tsa.stattools import adfuller
    out=pd.DataFrame(columns=['adfStat','pVal','lags','nObs','95% conf','corr'])
    #df0=pd.read_csv(path+instName+'.csv',index_col=0,parse_dates=True)
    df0 = data
    for d in np.linspace(0,1,11):
        df1=np.log(df0[['close']]).resample('1D').last() # downcast to daily obs
        df2=fracDiff_FFD(df1,d,thres=.01)
        corr=np.corrcoef(df1.loc[df2.index,'close'],df2['close'])[0,1]
        df2=adfuller(df2['close'],maxlag=1,regression='c',autolag=None)
        out.loc[d]=list(df2[:4])+[df2[4]['5%']]+[corr] # with critical value
    return out

def getMinFFD_df(data):
    from statsmodels.tsa.stattools import adfuller
    
    #df0=pd.read_csv(path+instName+'.csv',index_col=0,parse_dates=True)
    #df0 = data
    
    d_vals = np.linspace(0,1,11)
    iterables = [data.columns, d_vals]
    midx = pd.MultiIndex.from_product(iterables, names=['feature', 'd'])
    out=pd.DataFrame(columns=['adfStat','pVal','lags','nObs','95% conf','corr'], index=midx)
    for d in d_vals:
        #df1=np.log(df0[['close']]).resample('1D').last() # downcast to daily obs
        df2=fracDiff_FFD(data,d,thres=.01)
        
        for feature in df2.columns:
            #testNA = df2[feature].isnull().values.any()
            corr=np.corrcoef(data.loc[df2.index, feature],df2[feature])[0,1]
            feat_adf=adfuller(df2[feature],maxlag=1,regression='c',autolag=None)
            #out.loc[d]=list(df2[:4])+[df2[4]['5%']]+[corr] # with critical value
            out.loc[feature, d]=list(feat_adf[:4])+[feat_adf[4]['5%']] +[corr] # with critical value
    return out


def getMinFFD_features(data, dValRange=None):
    from statsmodels.tsa.stattools import adfuller
    
    #df0=pd.read_csv(path+instName+'.csv',index_col=0,parse_dates=True)
    #df0 = data
    if dValRange != None:
        d_vals = dValRange
    else:
        d_vals = np.linspace(0,1,11)
    out = pd.DataFrame(columns=data.columns, index=data.index)
    filled_columns = []
    feature_names = data.columns.tolist()

    for d in tqdm(d_vals):
        #df1=np.log(df0[['close']]).resample('1D').last() # downcast to daily obs
        df2=fracDiff_FFD(data,d,thres=.01)
        

        for feature in data.columns:
            #testNA = df2[feature].isnull().values.any()
            #corr=np.corrcoef(data.loc[df2.index, feature],df2[feature])[0,1]
            feat_adf=adfuller(df2[feature],maxlag=1,regression='c',autolag=None)
            if feat_adf[1] < 0.05:
                feat_val = df2[feature].copy()
                out.loc[feat_val.index, feature] = feat_val
                #out[feature] = df2[feature].copy()
                filled_columns.append(feature)
                data.drop(feature, axis=1, inplace=True)
        if data.empty:
            return out
    return out

def getMinFFD_feature_wise(data, dValRange=None):
    from statsmodels.tsa.stattools import adfuller
    
    #df0=pd.read_csv(path+instName+'.csv',index_col=0,parse_dates=True)
    #df0 = data
    if dValRange is not None:
        d_vals = dValRange
    else:
        d_vals = np.linspace(0,1,11)
    out = pd.DataFrame(columns=data.columns, index=data.index)
    filled_columns = []
    feature_names = data.columns.tolist()

    for feature in data.columns:
        
        #df1=np.log(df0[['close']]).resample('1D').last() # downcast to daily obs
        for d in tqdm(d_vals):
            df2=fracDiff_FFD(data[[feature]],d,thres=.01)
        
            #testNA = df2[feature].isnull().values.any()
            #corr=np.corrcoef(data.loc[df2.index, feature],df2[feature])[0,1]
            feat_adf=adfuller(df2[feature],maxlag=1,regression='c',autolag=None)
            if feat_adf[1] < 0.05:
                feat_val = df2[feature].copy()
                out.loc[feat_val.index, feature] = feat_val
                #out[feature] = df2[feature].copy()
                filled_columns.append(feature)
                #data.drop(feature, axis=1, inplace=True)
                break
        
        if data.empty:
            return out
    return out


def getMinFFD_feature_wise_dStar(data, dValRange=None):
    from statsmodels.tsa.stattools import adfuller
    dStarValues = {}

    #df0=pd.read_csv(path+instName+'.csv',index_col=0,parse_dates=True)
    #df0 = data
    if dValRange is not None:
        d_vals = dValRange
    else:
        d_vals = np.linspace(0,1,11)
    out = pd.DataFrame(columns=data.columns, index=data.index)
    filled_columns = []
    feature_names = data.columns.tolist()

    for feature in data.columns:
        
        #df1=np.log(df0[['close']]).resample('1D').last() # downcast to daily obs
        for d in tqdm(d_vals):
            df2=fracDiff_FFD(data[[feature]],d,thres=.01)
        
            #testNA = df2[feature].isnull().values.any()
            #corr=np.corrcoef(data.loc[df2.index, feature],df2[feature])[0,1]
            feat_adf=adfuller(df2[feature],maxlag=1,regression='c',autolag=None)
            if feat_adf[1] < 0.05:
                dStarValues[feature] = d
                feat_val = df2[feature].copy()
                out.loc[feat_val.index, feature] = feat_val
                #out[feature] = df2[feature].copy()
                filled_columns.append(feature)
                #data.drop(feature, axis=1, inplace=True)
                break
        
        if out[feature].empty:
            out.loc[feat_val.index, feature] = feat_val
    return out, dStarValues

def fracDiffFFD_dStar(data, dStar_values):
    dStarValues = {}
    out = pd.DataFrame(columns=data.columns, index=data.index)

    for feature, dStar in dStar_values.items():
        df2 = fracDiff_FFD(data[[feature]],dStar,thres=.01)

        feat_val = df2[feature].copy()
        out.loc[feat_val.index, feature] = feat_val
    return out

def getFeatureADF(data):
    adfFrame = pd.DataFrame(columns=['adfStat','pVal','lags','nObs','95% conf',], index=data.columns)
    for feature in data.columns:
        feat_adf=adfuller(data[feature],maxlag=1,regression='c',autolag=None)
        #out.loc[d]=list(df2[:4])+[df2[4]['5%']]+[corr] # with critical value
        adfFrame.loc[feature]=list(feat_adf[:4])+[feat_adf[4]['5%']]
    return adfFrame

def getNonStationaryFeats(data):
    nsFeats = []
    for feature in data.columns:
        feat_adf=adfuller(data[feature],maxlag=1,regression='c',autolag=None)
        if feat_adf[1] < 0.05:
            continue
        else:
            nsFeats.append(feature)

    return nsFeats

def getMinFFDPerFeature(downcastData):
    nstFeats = getNonStationaryFeats(downcastData)
    minD = {key : 0 for key in nstFeats}
    d_vals = np.linspace(0,1,11)


    for feature in tqdm(nstFeats):
        featureSeries = downcastData[[feature]]

        for d in d_vals:
            fdSeries=fracDiff_FFD(featureSeries,d,thres=.01)
            
            feat_adf=adfuller(fdSeries[feature],maxlag=1,regression='c',autolag=None)
            if feat_adf[1] < 0.05:
                minD[feature] = d
                break

    return minD


def getDataFFD(data, dValDict):
    out = pd.DataFrame(columns=data.columns, index=data.index)

    for feature in data.columns:
        featureSeries = data[[feature]]

        ffdSeries = fracDiff_FFD(featureSeries, dValDict[feature])
        out[feature] = ffdSeries

    return out

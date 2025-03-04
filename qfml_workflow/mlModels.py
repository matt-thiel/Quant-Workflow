from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from qfml_workflow import crossValidation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib as ta

#https://www.scitepress.org/Papers/2020/94262/94262.pdf


class PCARenamer(BaseEstimator, TransformerMixin):
    def __init__(self, nCat, catFeats):
        #self.n_components = n_components
        self.nCat = nCat 
        self.catFeats = catFeats


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(X.T[1])
        pca_columns = [f'PCA_{i}' for i in range(1, X.shape[1] - self.nCat + 1)]
        outFeats = pca_columns+self.catFeats
        return pd.DataFrame(X, columns=outFeats)
    
class StandardRenamer(BaseEstimator, TransformerMixin):
    def __init__(self, catFeats, contFeats):
        #self.n_components = n_components
        self.catFeats = catFeats
        self.contFeats = contFeats


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.catFeats+self.contFeats)


class ModelsML:
    def __init__(self, continuousFeats, categoricalFeats):
        self.baseModels = []
        self.pipelinesPCA = []
        self.pipelinesStandard = []

        rfModel = RandomForestClassifier(n_estimators=400,
                                        criterion='gini',
                                        max_depth=None,
                                        max_features=1,
                                        class_weight='balanced_subsample',
                                        min_weight_fraction_leaf=0.05,
                                        random_state = 12
                                        )

        adaModel = AdaBoostClassifier(n_estimators=400,
                                    learning_rate=1,
                                    random_state = 12)

        svcModel = SVC(gamma='auto', cache_size=200, 
                    decision_function_shape='ovr',
                    probability=True, class_weight='balanced',
                    random_state = 12)

        xgbModel = XGBClassifier(eta=1, max_depth=3,
                                use_label_encoder=False,eval_metric='logloss',
                                min_child_weight=1, learning_rate=0.02,
                                random_state = 12)

        gnbModel = GaussianNB(priors='None', var_smoothing=1e-9, )

        cnbModel = ComplementNB()

        knnModel = KNeighborsClassifier(algorithm='auto',n_neighbors=5)

        baggingCLFRF = RandomForestClassifier(n_estimators=1, criterion='entropy', 
                                            bootstrap=False, class_weight='balanced_subsample')
        # make sure to set max_samples to avgU
        #baggingRFModel = BaggingClassifier(base_estimator=baggingCLFRF, n_estimators=1000, max_features=1.,
        baggingRFModel = BaggingClassifier(base_estimator=baggingCLFRF, n_estimators=400, max_features=1.,
                                            random_state = 12)

        baggingCLFDT = DecisionTreeClassifier(criterion='entropy', 
                                            max_features='auto',
                                            class_weight='balanced')

        #baggingDTModel = BaggingClassifier(base_estimator=baggingCLFDT, n_estimators=1000, max_features=1,
        baggingDTModel = BaggingClassifier(base_estimator=baggingCLFDT, n_estimators=400, max_features=1,
                                                random_state = 12)
        

        #models.append(('kNN', knnModel))
        #models.append(('cNB', cnbModel))
        self.baseModels.append(('RF', rfModel))
        self.baseModels.append(('ADA', adaModel))
        self.baseModels.append(('SVC', svcModel))
        #models.append(('gNB', gnbModel))
        self.baseModels.append(('XGB', xgbModel))
        self.baseModels.append(('bgRF', baggingRFModel))
        self.baseModels.append(('bgDT', baggingDTModel))

        for modelName, model in self.baseModels:
            basePCAPipe = Pipeline(steps=[
                ('Scaler', StandardScaler()),
                ('PCA', PCA(n_components=0.95))
            ])
            pcaModel = clone(model)
            pcaPipe = Pipeline(steps=[
                ("scaleCT", ColumnTransformer(
                    [
                        ('pcaPipe', basePCAPipe, continuousFeats),
                        ('pass', 'passthrough', categoricalFeats),
                    ]
                )),
                #('RFE_'+modelName, RFE(DecisionTreeClassifier(class_weight='balanced', criterion='entropy'),)),
                #('pre-model', baggingRFModel),
                #('modelSelect', SelectFromModel(rfModel, prefit=False, threshold='0.5*mean')),
                
                (modelName, pcaModel)
            ])

            standardModel = clone(model)
            standardPipe = Pipeline(steps=[
                ("scaleCT", ColumnTransformer(
                    [
                        ('Scaler', StandardScaler(), continuousFeats),
                        ('pass', 'passthrough', categoricalFeats),
                    ]
                )),
                ('stdRenamer', StandardRenamer(categoricalFeats, continuousFeats)),
                #('RFE_'+modelName, RFE(DecisionTreeClassifier(class_weight='balanced', criterion='entropy'),)),
                #('pre-model', baggingRFModel),
                #('modelSelect', SelectFromModel(rfModel, prefit=False, threshold='0.5*mean')),
                (modelName, standardModel)
            ])
                                #'+ _pipe_PCA'
            self.pipelinesPCA.append((modelName , modelName, pcaPipe))
            self.pipelinesStandard.append((modelName + '_pipe_standard', modelName, standardPipe))


    

def selectFeatures(model, cvMethod, scoreMethod):
    pass

def selectModelPipeline(pipeList, X, y, tbEvents, sampleWeight, avgUMean, 
                        numFolds, cvGen, pctEmbargo=0.01, scoringMethod='neg_log_loss'):
    results = []
    modelNames = []

    for pipeName, modelName, pipeModel in pipeList:
        if modelName == 'bgRF' or modelName== 'bgDT':
            pipeModel[modelName].set_params(max_samples = avgUMean)

        cvResults = crossValidation.cvScoreCustomPipe(pipeModel, X, y, sample_weight=sampleWeight, 
                                pipeSampleParam=modelName+'__sample_weight',scoring=scoringMethod, 
                                cv=numFolds, t1= tbEvents['t1'],
                                cvGen = cvGen, pctEmbargo=pctEmbargo)
        results.append(cvResults)
        modelNames.append(pipeName)
        print(print('%s: %f (%f)' % (pipeName, cvResults.mean(), cvResults.std())))

    plt.boxplot(results, labels=modelNames)
    plt.title(f'Algorithm Comparison, Scoring={scoringMethod}')
    plt.show()

# model selection should be done on train set
def selectModel(modelList, X, y, tbEvents, sampleWeight, avgUniqueness, 
                folds, cvMethod, pctEmbargo=0.01, scoringMethod='neg_log_loss',
                scaler=None,pca=None):
    results = []
    modelNames = []

    for name, model in modelList:
        # may want to change
        if name == 'bgRF' or name== 'bgDT':
            model.set_params(max_samples = avgUniqueness)

        if (scaler is not None) and (pca is not None):
            processedModel = Pipeline(steps=[scaler,
                                    pca,
                                    (name, model)])

            cvResults = crossValidation.cvScoreCustomPipe(processedModel, X, y, sample_weight=sampleWeight, 
                                pipeSampleParam=name+'__sample_weight',scoring=scoringMethod, 
                                cv=folds, t1= tbEvents['t1'],
                                cvGen = cvMethod, pctEmbargo=pctEmbargo)
        elif (scaler is not None) and (pca is None):
            processedModel = Pipeline(steps=[("scaler", scaler),
                                    (name, model)])
            
            cvResults = crossValidation.cvScoreCustomPipe(processedModel, X, y, sample_weight=sampleWeight, 
                                pipeSampleParam=name+'__sample_weight', scoring=scoringMethod, 
                                cv=folds, t1= tbEvents['t1'],
                                cvGen = cvMethod, pctEmbargo=pctEmbargo)
        else:
            processedModel = model

            cvResults = crossValidation.cvScore(processedModel, X, y, sample_weight=sampleWeight, 
                                scoring=scoringMethod, cv=folds, t1= tbEvents['t1'],
                                cvGen = cvMethod, pctEmbargo=pctEmbargo)
        results.append(cvResults)
        modelNames.append(name)
        print(print('%s: %f (%f)' % (name, cvResults.mean(), cvResults.std())))

    plt.boxplot(results, labels=modelNames)
    plt.title('Algorithm Comparison')
    plt.show()


def createFeaturesoLD(dataset):
    # volume
    dataset['obv'] = ta.OBV(dataset['close'], dataset['volume'])
    dataset['volume_ewa_20'] = ta.EMA(dataset['volume'], timeperiod=20)

 

    nonFeatures = ['open', 'high', 'low', 'close', 'tick_num', 'num_ticks_sampled','fast_mavg', 'slow_mavg', 'side']
    featureList = [x for x in dataset.columns if x not in nonFeatures]

    # control for lookahead biasS
    dataset[featureList] = dataset[featureList].shift(1)
    #secondaryModelData = secondaryModelData.dropna()

    return dataset, featureList, nonFeatures

#def createFeaturesLong(dataset):
 #   dataset['volatility_50'] = dataset['log_ret'].rolling(window=50, min_periods=50, center=False).std()
 #   dataset['volatility_25'] = dataset['log_ret'].rolling(window=25, min_periods=100, center=False).std()
 #   dataset['volatility_10'] = dataset['log_ret'].rolling(window=10, min_periods=250, center=False).std()

from qfml_workflow import fractionalDifferencing

def createComboFeatures(dataset):

    #dataset['ffdClose'] = fractionalDifferencing.getMinFFD_feature_wise(dataset[['close']])
    #dataset['ffdClose'] = pd.to_numeric(dataset['ffdClose'])
    #dataset['log_ret'] = np.log(dataset['ffdClose'])
    #dataset = dataset.drop('ffdClose', axis=1)

    #contFeatures, featureList, nonFeatures = paperFeaturesContinuous(dataset)
    dataset, featureListCont, nonFeaturesCont = featuresContinuousMultiParam(dataset, [5,12,25],)
    # Log Returns
    dataset['log_ret'] = np.log(dataset['close']).diff()

    # Momentum
    dataset['mom1'] = dataset['close'].pct_change(periods=1)
    dataset['mom2'] = dataset['close'].pct_change(periods=2)
    dataset['mom3'] = dataset['close'].pct_change(periods=3)
    dataset['mom4'] = dataset['close'].pct_change(periods=4)
    dataset['mom5'] = dataset['close'].pct_change(periods=5)

    # Volatility
    dataset['volatility_50'] = dataset['log_ret'].rolling(window=50, min_periods=50, center=False).std()
    dataset['volatility_31'] = dataset['log_ret'].rolling(window=31, min_periods=31, center=False).std()
    dataset['volatility_15'] = dataset['log_ret'].rolling(window=15, min_periods=15, center=False).std()

    # Serial Correlation (Takes about 4 minutes)
    window_autocorr = 50

    dataset['autocorr_1'] = dataset['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=1), raw=False)
    dataset['autocorr_2'] = dataset['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=2), raw=False)
    dataset['autocorr_3'] = dataset['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=3), raw=False)
    dataset['autocorr_4'] = dataset['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=4), raw=False)
    dataset['autocorr_5'] = dataset['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=5), raw=False)

    # Get the various log -t returns
    dataset['log_t1'] = dataset['log_ret'].shift(1)
    dataset['log_t2'] = dataset['log_ret'].shift(2)
    dataset['log_t3'] = dataset['log_ret'].shift(3)
    dataset['log_t4'] = dataset['log_ret'].shift(4)
    dataset['log_t5'] = dataset['log_ret'].shift(5)


    #dataset = pd.concat([dataset, contFeatures],axis=1)

    #dataset['sma_10_trend'] = np.where(contFeatures['sma_10'] < priceFrame['close'], 1, -1)

    #dataset['wma_10_trend'] = np.where(contFeatures['wma_10'] < dataset['close'], 1, -1)

    #dataset['sma_10_trend'] = np.where(contFeatures['sma_10'] < dataset['close'], 1, -1)

    #dataset['wma_10_trend'] = np.where(contFeatures['wma_10'] < dataset['close'], 1, -1)

    #dataset['mom_10_trend'] = np.where(contFeatures['mom_10'] > 0, 1, -1)

    #dataset['willR_10_trend'] = np.where(contFeatures['willR_10'] > contFeatures['willR_10'].shift(1), 1, -1)

    # can use macdhist too
    #dataset['macd_trend'] = np.where(contFeatures['macd'] > contFeatures['macd'].shift(1), 1, -1)

    #dataset['stoch_k_trend'] = np.where(contFeatures['stoch_k'] > contFeatures['stoch_k'].shift(1), 1, -1)
    #dataset['stoch_d_trend'] = np.where(contFeatures['stoch_d'] > contFeatures['stoch_d'].shift(1), 1, -1)

    #rsiConditions = [contFeatures['rsi_10'] > 70,
    #                 (contFeatures['rsi_10'] <= 70) & (contFeatures['rsi_10'] >= 30) & 
     #                (contFeatures['rsi_10'] > contFeatures['rsi_10'].shift(1)),
     #                (contFeatures['rsi_10'] <= 70) & (contFeatures['rsi_10'] >= 30) & 
     #                (contFeatures['rsi_10'] < contFeatures['rsi_10'].shift(1)),
     #                contFeatures['rsi_10'] < 30]
    #rsiChoices = [1, 1, -1, -1]
   # dataset['rsi_10_trend'] = np.select(rsiConditions, rsiChoices)

    #cciConditions = [contFeatures['CCI_10'] > 200,
      #               contFeatures['CCI_10'] < -200,]
    #cciChoices = [-1, 1]
    #dataset['CCI_10_trend'] = np.select(cciConditions, cciChoices)

    #dataset['ad_osc_trend'] = np.where(contFeatures['ad_osc'] > contFeatures['ad_osc'].shift(1), 1, -1)

    nonFeatures = ['open', 'high', 'low', 'close', 'volume', 'tick_num', 'num_ticks_sampled','fast_mavg', 'slow_mavg', 'side']
    featureList = [x for x in dataset.columns if x not in nonFeatures]
    print(len(featureList))
    #categoricalFeats = ['rsi_10_trend', 'stoch_d_trend', 'stoch_k_trend', 'mom_10_trend', 
    #                    'wma_10_trend']
    categoricalFeats = []
    print(len(dataset.columns))

    # control for lookahead bias
    dataset[featureList] = dataset[featureList].shift(1)
    #secondaryModelData = secondaryModelData.dropna()
    
    continuousFeats = [x for x in featureList if x not in categoricalFeats]

    return dataset, featureList, categoricalFeats, continuousFeats, nonFeatures

def createFeaturesHT(dataset):

    dataset['ffdClose'] = fractionalDifferencing.getMinFFD_feature_wise(dataset[['close']])
    dataset['ffdClose'] = pd.to_numeric(dataset['ffdClose'])
    dataset['log_ret'] = np.log(dataset['ffdClose'])
    dataset = dataset.drop('ffdClose', axis=1)
    # Log Returns
    #dataset['log_ret'] = np.log(dataset['close']).diff()

    # Momentum
    dataset['mom1'] = dataset['close'].pct_change(periods=1)
    dataset['mom2'] = dataset['close'].pct_change(periods=2)
    dataset['mom3'] = dataset['close'].pct_change(periods=3)
    dataset['mom4'] = dataset['close'].pct_change(periods=4)
    dataset['mom5'] = dataset['close'].pct_change(periods=5)

    # Volatility
    dataset['volatility_50'] = dataset['log_ret'].rolling(window=50, min_periods=50, center=False).std()
    dataset['volatility_31'] = dataset['log_ret'].rolling(window=31, min_periods=31, center=False).std()
    dataset['volatility_15'] = dataset['log_ret'].rolling(window=15, min_periods=15, center=False).std()

    # Serial Correlation (Takes about 4 minutes)
    window_autocorr = 50

    dataset['autocorr_1'] = dataset['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=1), raw=False)
    dataset['autocorr_2'] = dataset['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=2), raw=False)
    dataset['autocorr_3'] = dataset['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=3), raw=False)
    dataset['autocorr_4'] = dataset['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=4), raw=False)
    dataset['autocorr_5'] = dataset['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=5), raw=False)

    # Get the various log -t returns
    dataset['log_t1'] = dataset['log_ret'].shift(1)
    dataset['log_t2'] = dataset['log_ret'].shift(2)
    dataset['log_t3'] = dataset['log_ret'].shift(3)
    dataset['log_t4'] = dataset['log_ret'].shift(4)
    dataset['log_t5'] = dataset['log_ret'].shift(5)

    nonFeatures = ['open', 'high', 'low', 'close', 'volume', 'tick_num', 'num_ticks_sampled','fast_mavg', 'slow_mavg', 'side']
    featureList = [x for x in dataset.columns if x not in nonFeatures]

    # control for lookahead bias
    dataset[featureList] = dataset[featureList].shift(1)
    #secondaryModelData = secondaryModelData.dropna()

    return dataset, featureList, nonFeatures


def createFeatures(dataset):
    # Example features to use
    #featureList = ['log_ret', 'mom1', 'mom2', 'mom3', 'mom4', 'mom5', 
    #                'volatility_10', 'volatility_25', 'volatility_50', 
    #               'log_t1', 'log_t2', 'log_t3', 'log_t4', 'log_t5', 'volume']

    #featureList = [ 'mom3',  'mom5', 
    #               'volatility_25', 'volatility_50', 
    #              'log_t1', 'log_t5', ]
    #fracDiffCLose, dStarClose = fractionalDifferencing.getMinFFD_feature_wise_dStar(dollarBars[['close']])
    #fradDiffClose = fracDiffCLose.apply(pd.to_numeric)
    dataset['log_ret'] = np.log(dataset['close']).diff()#secondaryModelData['close'].diff()
    #dollarBars['log_ret'] = np.log(fradDiffClose)

    # momentum
    dataset['mom5'] = dataset['close'].pct_change(periods=50)

    #dataset['fracDiff_mom1'] = fradDiffClose.shift(1)
    #dataset['fracDiff_mom2'] = fradDiffClose.shift(2)
    #dataset['fracDiff_mom3'] = fradDiffClose.shift(3)
    #dataset['fracDiff_mom4'] = fradDiffClose.shift(4)
    #dataset['fracDiff_mom5'] = fradDiffClose.shift(5)
    # Volatility
    dataset['volatility_50'] = dataset['log_ret'].rolling(window=50, min_periods=50, center=False).std()
    dataset['volatility_100'] = dataset['log_ret'].rolling(window=100, min_periods=50, center=False).std()
    dataset['volatility_250'] = dataset['log_ret'].rolling(window=250, min_periods=50, center=False).std()

    dataset[f'trix_{5}'] = ta.TRIX(dataset['close'], timeperiod=5)
    dataset[f'trix_{12}'] = ta.TRIX(dataset['close'], timeperiod=12)
    dataset[f'trix_{26}'] = ta.TRIX(dataset['close'], timeperiod=26)


    # Serial Correlation (Takes about 4 minutes)
    window_autocorr = 50

    dataset['autocorr_1'] = dataset['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=1), raw=False)
    #dataset['autocorr_2'] = dataset['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=2), raw=False)
    dataset['autocorr_3'] = dataset['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=3), raw=False)
    #dataset['autocorr_4'] = dataset['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=4), raw=False)
    dataset['autocorr_5'] = dataset['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=5), raw=False)
    # 
    # Momentum
    #dataset['moneyFlow_14'] = ta.MFI(dataset['high'], dataset['low'], dataset['close'], dataset['volume'], timeperiod=14)
    dataset['rsi_50'] = ta.RSI(dataset['close'], timeperiod=50)
    #dataset['rsi_signal'] = 0
    #dataset['rsi_signal'].loc[dataset['rsi_14'] > 70] = -1 
    #dataset['rsi_signal'].loc[dataset['rsi_14'] < 30] = 1 
    dataset['adx_50'] = ta.ADX(dataset['high'], dataset['low'], dataset['close'], timeperiod=50)
    #dataset['adx_signal'] = 0
    #dataset['adx_signal'].loc[dataset['adx_14'] >= 20] = 1

    dataset['return_ewa_50'] = ta.EMA(dataset['log_ret'], timeperiod=20)
    dataset['return_ewa_50'] = ta.EMA(dataset['log_ret'], timeperiod=20)
    # overlap
    #bband width?

    dataset['stoch_k'], dataset['stoch_d'] = ta.STOCH(dataset['high'], dataset['low'], dataset['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    # volume
    dataset['obv'] = ta.OBV(dataset['close'], dataset['volume'])
    dataset['volume_ewa_20'] = ta.EMA(dataset['volume'], timeperiod=20)
    dataset['volume_ewa_500'] = ta.EMA(dataset['volume'], timeperiod=50)
    dataset['volume_ewa_100'] = ta.EMA(dataset['volume'], timeperiod=100)

    # Volatility
    #dataset['atr_14'] = ta.ATR(dataset['high'], dataset['low'], dataset['close'], timeperiod=14)

    # cycle ind
    #dataset['hilbert_transform'] = ta.HT_TRENDMODE(dataset['close'])

    # math funcs
    #dataset['beta_50'] = ta.BETA(dataset['high'], dataset['low'], timeperiod=20)

    nonFeatures = ['open', 'high', 'low', 'close', 'volume', 'tick_num', 'num_ticks_sampled','fast_mavg', 'slow_mavg', 'side']
    featureList = [x for x in dataset.columns if x not in nonFeatures]

    # control for lookahead bias
    dataset[featureList] = dataset[featureList].shift(1)
    #secondaryModelData = secondaryModelData.dropna()

    return dataset, featureList, nonFeatures

def customFeaturesNew(dataset, paramDict):
    # RSI14
    for param in paramDict:
        rsiConditions = [contFeatures['rsi_10'] > 70,
                        (contFeatures['rsi_10'] <= 70) & (contFeatures['rsi_10'] >= 30) & 
                        (contFeatures['rsi_10'] > contFeatures['rsi_10'].shift(1)),
                        (contFeatures['rsi_10'] <= 70) & (contFeatures['rsi_10'] >= 30) & 
                        (contFeatures['rsi_10'] < contFeatures['rsi_10'].shift(1)),
                        contFeatures['rsi_10'] < 30]
        rsiChoices = [1, 1, -1, -1]
        dataset['rsi_10_trend'] = np.select(rsiConditions, rsiChoices)

        dataset['ad_osc_trend'] = np.where(contFeatures['ad_osc'] > contFeatures['ad_osc'].shift(1), 1, -1)


        dataset[f'trix_{param}_trend'] = np.where(contFeatures[f'trix_{param}']>0, 1, -1)

        dataset[f'bbandDiff_{param}'] = contFeatures[f'upperband_ffd_{param}'] -contFeatures[f'lowerband_ffd_{param}']
        dataset[f'bbandUpper_{param}_trend'] = np.where(dataset['ffdClose'] > contFeatures[f'upperband_ffd_{param}'], 1, 0)
        dataset[f'bbandLower_{param}_trend'] = np.where(dataset['ffdClose'] < contFeatures[f'lowerband_ffd_{param}'], -1, 0)

        dataset[f'aroon_osc_{param}_trend'] = np.where(contFeatures[f'aroon_osc_{param}'] > 0, 1, -1)


def paperFeaturesContinuous(dataset, ffdPrices=None):
    priceColumns = ['open', 'high','low','close','volume']
    if ffdPrices is not None:
        priceFrame = ffdPrices
    else:
        priceFrame = dataset[priceColumns]
    
    dataset['sma_10'] = priceFrame['close'].rolling(10).mean()

    dataset['wma_10'] = ta.WMA(priceFrame['close'], timeperiod=10)

    dataset['mom_10'] = ta.MOM(priceFrame['close'], timeperiod=10)

    dataset['willR_10'] = ta.WILLR(priceFrame['high'], priceFrame['low'], priceFrame['close'], timeperiod=14)

    # can use macdhist too
    dataset['macd'], _, _ = ta.MACD(priceFrame['close'], fastperiod=12, slowperiod=26, signalperiod=9)

    dataset['stoch_k'], dataset['stoch_d'] = ta.STOCH(priceFrame['high'], priceFrame['low'], priceFrame['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    dataset['rsi_10'] = ta.RSI(priceFrame['close'], timeperiod=10)

    dataset['CCI_10'] = ta.CCI(priceFrame['high'], priceFrame['low'], priceFrame['close'], timeperiod=10)

    dataset['ad_osc'] = (priceFrame['high'] - priceFrame['close'].shift(1)) / (priceFrame['high'] - priceFrame['low'])

    nonFeatures = ['open', 'high', 'low', 'close', 'tick_num', 'num_ticks_sampled','fast_mavg', 'slow_mavg', 'side']
    featureList = [x for x in dataset.columns if x not in nonFeatures]

    # control for lookahead bias
    dataset[featureList] = dataset[featureList].shift(1)
    #secondaryModelData = secondaryModelData.dropna()

    return dataset, featureList, nonFeatures

def paperFeaturesTrendLayer(dataset, ffdPrices=None):
    priceColumns = ['open', 'high','low','close','volume']
    if ffdPrices is not None:
        priceFrame = ffdPrices
    else:
        priceFrame = dataset[priceColumns]

    contFeatures, featureList, nonFeatures = paperFeaturesContinuous(priceFrame)

    #dataset['sma_10_trend'] = np.where(contFeatures['sma_10'] < priceFrame['close'], 1, -1)

    #dataset['wma_10_trend'] = np.where(contFeatures['wma_10'] < priceFrame['close'], 1, -1)

    dataset['sma_10_trend'] = np.where(contFeatures['sma_10'] < priceFrame['close'], 1, -1)

    dataset['wma_10_trend'] = np.where(contFeatures['wma_10'] < priceFrame['close'], 1, -1)

    dataset['mom_10_trend'] = np.where(contFeatures['mom_10'] > 0, 1, -1)

    dataset['willR_10_trend'] = np.where(contFeatures['willR_10'] > contFeatures['willR_10'].shift(1), 1, -1)

    # can use macdhist too
    dataset['macd_trend'] = np.where(contFeatures['macd'] > contFeatures['macd'].shift(1), 1, -1)

    dataset['stoch_k_trend'] = np.where(contFeatures['stoch_k'] > contFeatures['stoch_k'].shift(1), 1, -1)
    dataset['stoch_d_trend'] = np.where(contFeatures['stoch_d'] > contFeatures['stoch_d'].shift(1), 1, -1)

    rsiConditions = [contFeatures['rsi_10'] > 70,
                     (contFeatures['rsi_10'] <= 70) & (contFeatures['rsi_10'] >= 30) & 
                     (contFeatures['rsi_10'] > contFeatures['rsi_10'].shift(1)),
                     (contFeatures['rsi_10'] <= 70) & (contFeatures['rsi_10'] >= 30) & 
                     (contFeatures['rsi_10'] < contFeatures['rsi_10'].shift(1)),
                     contFeatures['rsi_10'] < 30]
    rsiChoices = [1, 1, -1, -1]
    dataset['rsi_10_trend'] = np.select(rsiConditions, rsiChoices)

    cciConditions = [contFeatures['CCI_10'] > 200,
                     contFeatures['CCI_10'] < -200,]
    cciChoices = [-1, 1]
    dataset['CCI_10_trend'] = np.select(cciConditions, cciChoices)

    dataset['ad_osc_trend'] = np.where(contFeatures['ad_osc'] > contFeatures['ad_osc'].shift(1), 1, -1)


    nonFeatures = ['open', 'high', 'low', 'close', 'tick_num', 'num_ticks_sampled','fast_mavg', 'slow_mavg', 'side']
    featureList = [x for x in dataset.columns if x not in nonFeatures]

    # control for lookahead bias
    dataset[featureList] = dataset[featureList].shift(1)
    #secondaryModelData = secondaryModelData.dropna()

    return dataset, featureList, nonFeatures

def featuresContinuousMultiParam(dataset, paramDict, ffdPrices=None):
    priceColumns = ['open', 'high','low','close','volume']
    if ffdPrices is not None:
        priceFrame = ffdPrices
    else:
        priceFrame = dataset[priceColumns]

    for param in paramDict:
        dataset[f'sma_{param}'] = priceFrame['close'].rolling(param).mean()

        dataset[f'wma_{param}'] = ta.WMA(priceFrame['close'], timeperiod=param)

        dataset[f'mom_{param}'] = ta.MOM(priceFrame['close'], timeperiod=param)

        dataset[f'willR_{param}'] = ta.WILLR(priceFrame['high'], priceFrame['low'], priceFrame['close'], timeperiod=param)

        # can use macdhist too
        dataset[f'macd_{param}'], _, _ = ta.MACD(priceFrame['close'], fastperiod=param//2, slowperiod=param, signalperiod=param//(3/4))

        dataset[f'stoch_k_{param}'], dataset[f'stoch_d_{param}'] = ta.STOCH(priceFrame['high'], priceFrame['low'], priceFrame['close'], fastk_period=param, slowk_period=param//2, slowk_matype=0, slowd_period=param//2, slowd_matype=0)

        dataset[f'rsi_{param}'] = ta.RSI(priceFrame['close'], timeperiod=param)

        dataset[f'CCI_{param}'] = ta.CCI(priceFrame['high'], priceFrame['low'], priceFrame['close'], timeperiod=param)

        dataset[f'ad_osc_{param}'] = (priceFrame['high'] - priceFrame['close'].shift(param)) / (priceFrame['high'] - priceFrame['low'])

    nonFeatures = ['open', 'high', 'low', 'close', 'tick_num', 'num_ticks_sampled','fast_mavg', 'slow_mavg', 'side']
    featureList = [x for x in dataset.columns if x not in nonFeatures]

    # control for lookahead bias
    dataset[featureList] = dataset[featureList].shift(1)
    #secondaryModelData = secondaryModelData.dropna()

    return dataset, featureList, nonFeatures


def paperFeaturesTrendLayerMultiParam(dataset, paramDict, ffdPrices=None):
    priceColumns = ['open', 'high','low','close','volume']
    if ffdPrices is not None:
        priceFrame = ffdPrices
    else:
        priceFrame = dataset[priceColumns]

    contFeatures, featureList, nonFeatures = featuresContinuousMultiParam(priceFrame, paramDict)

    #dataset['sma_10_trend'] = np.where(contFeatures['sma_10'] < priceFrame['close'], 1, -1)

    #dataset['wma_10_trend'] = np.where(contFeatures['wma_10'] < priceFrame['close'], 1, -1)
    for param in paramDict:

        dataset[f'sma_{param}_trend'] = np.where(contFeatures[f'sma_{param}'] < priceFrame['close'], 1, -1)

        dataset[f'wma_{param}_trend'] = np.where(contFeatures[f'wma_{param}'] < priceFrame['close'], 1, -1)

        dataset[f'mom_{param}_trend'] = np.where(contFeatures[f'mom_{param}'] > 0, 1, -1)

        dataset[f'willR_{param}_trend'] = np.where(contFeatures[f'willR_{param}'] > contFeatures[f'willR_{param}'].shift(1), 1, -1)

        # can use macdhist too
        dataset[f'macd_{param}_trend'] = np.where(contFeatures[f'macd_{param}'] > contFeatures[f'macd_{param}'].shift(1), 1, -1)

        dataset[f'stoch_k_{param}_trend'] = np.where(contFeatures[f'stoch_k_{param}'] > contFeatures[f'stoch_k_{param}'].shift(1), 1, -1)
        dataset[f'stoch_d_{param}_trend'] = np.where(contFeatures[f'stoch_k_{param}'] > contFeatures[f'stoch_d_{param}'].shift(1), 1, -1)

        rsiConditions = [contFeatures[f'rsi_{param}'] > 70,
                        (contFeatures[f'rsi_{param}'] <= 70) & (contFeatures[f'rsi_{param}'] >= 30) & 
                        (contFeatures[f'rsi_{param}'] > contFeatures[f'rsi_{param}'].shift(1)),
                        (contFeatures[f'rsi_{param}'] <= 70) & (contFeatures[f'rsi_{param}'] >= 30) & 
                        (contFeatures[f'rsi_{param}'] < contFeatures[f'rsi_{param}'].shift(1)),
                        contFeatures[f'rsi_{param}'] < 30]
        rsiChoices = [-1, 1, -1, 1]
        dataset[f'rsi_{param}_trend'] = np.select(rsiConditions, rsiChoices)

        cciConditions = [contFeatures[f'CCI_{param}'] > 200,
                        contFeatures[f'CCI_{param}'] < -200,]
        cciChoices = [-1, 1]
        #dataset[f'CCI_{param}_trend'] = np.select(cciConditions, cciChoices)

        dataset[f'ad_osc_{param}_trend'] = np.where(contFeatures[f'ad_osc_{param}'] > contFeatures[f'ad_osc_{param}'].shift(1), 1, -1)

        #dataset['ad_line'] = ta.SMA(ta.AD(dataset['high'], dataset['low'], dataset['close'], dataset['volume']), timeperiod=param)
        #adOsc_conditions = [(dataset['ad_line'] > dataset['ad_line'].shift(1)) &
        #                    (contFeatures[f'sma_{param}'] > contFeatures[f'sma_{param}'].shift(1)),
        #                    (dataset['ad_line'] < dataset['ad_line'].shift(1)) &
        #                    (contFeatures[f'sma_{param}'] < contFeatures[f'sma_{param}'].shift(1)),
         #                   (dataset['ad_line'] > dataset['ad_line'].shift(1)) &
         #                   (contFeatures[f'sma_{param}'] < contFeatures[f'sma_{param}'].shift(1)),
        #                   (dataset['ad_line'] < dataset['ad_line'].shift(1)) &
         #                   (contFeatures[f'sma_{param}'] > contFeatures[f'sma_{param}'].shift(1)),]
        #adOsc_choices = [1,-1,1,-1]
       # dataset[f'ad_osc_{param}_trend'] = np.select(adOsc_conditions,
       #                                              adOsc_choices)
        #dataset = dataset.drop('ad_line',axis=1)


    nonFeatures = ['open', 'high', 'low', 'close', 'tick_num', 'num_ticks_sampled','fast_mavg', 'slow_mavg', 'side']
    featureList = [x for x in dataset.columns if x not in nonFeatures]

    # control for lookahead bias
    dataset[featureList] = dataset[featureList].shift(1)
    #secondaryModelData = secondaryModelData.dropna()

    return dataset, featureList, nonFeatures

def allFeatures(datasetIn, paramDict):
    dataset = datasetIn.copy()
    for param in paramDict:
        window_autocorr =50
        dataset[f'autocorr_{param}'] = dataset['close'].rolling(window=param*2, min_periods=param*2, center=False).apply(lambda x: x.autocorr(lag=param), raw=False)
        dataset[f'autocorr_ffd_{param}'] = dataset['ffdClose'].rolling(window=param*2, min_periods=param//2, center=False).apply(lambda x: x.autocorr(lag=param//2), raw=False)
        dataset[f'volatility_ffd_{param}']  = dataset['ffdClose'].rolling(window=param, min_periods=param//2, center=False).std()
        dataset[f'upperband_ffd_{param}'] , dataset[f'middleband_ffd_{param}'], dataset[f'lowerband_ffd_{param}'] = ta.BBANDS(dataset['ffdClose'], timeperiod=param, nbdevup=2, nbdevdn=2, matype=0)
        dataset[f'sma_ffd_{param}'] = ta.SMA(dataset['ffdClose'], timeperiod=param)
        dataset[f'wma_ffd_{param}'] = ta.WMA(dataset['ffdClose'], timeperiod=param)
        dataset[f'ema_ffd_{param}'] = ta.EMA(dataset['ffdClose'], timeperiod=param)

        dataset[f'willR_{param}'] = ta.WILLR(dataset['high'], dataset['low'], dataset['close'], timeperiod=param)

        # can use macdhist too
        dataset[f'macd_{param}'], dataset[f'macd_signal_{param}'], dataset[f'macd_hist_{param}'] = ta.MACD(dataset['close'], fastperiod=param//2, slowperiod=param, signalperiod=((param//2) *(3/4)))

        dataset[f'stoch_k_{param}'], dataset[f'stoch_d_{param}'] = ta.STOCH(dataset['high'], dataset['low'], dataset['close'], fastk_period=param, slowk_period=param//2, slowk_matype=0, slowd_period=param//2, slowd_matype=0)

        dataset[f'rsi_{param}'] = ta.RSI(dataset['close'], timeperiod=param)

        dataset[f'CCI_{param}'] = ta.CCI(dataset['high'], dataset['low'], dataset['close'], timeperiod=param)

        dataset[f'ad_osc_{param}'] = (dataset['high'] - dataset['close'].shift(param)) / (dataset['high'] - dataset['low'])

        dataset[f'ad_osc_ffd_{param}'] = (dataset['ffdHigh'] / dataset['ffdClose'].shift(param)) / (dataset['ffdHigh'] / dataset['ffdLow'])

        dataset[f'atr_{param}'] = ta.ATR(dataset['high'], dataset['low'], dataset['close'], timeperiod=14)

        dataset[f'rsi_{param}'] = ta.RSI(dataset['close'], timeperiod=param)
        dataset[f'mom_{param}'] = ta.MOM(dataset['close'], timeperiod=param)

        # if varaince outside threshold?
        dataset[f'variance_ffd_{param}'] = ta.VAR(dataset['ffdClose'], timeperiod=param, nbdev=1)
        dataset[f'ppo_{param}'] = ta.PPO(dataset['close'], fastperiod=param//2, slowperiod=param, matype=0)
        dataset[f'dmi_{param}'] = ta.DX(dataset['high'], dataset['low'], dataset['close'], timeperiod=param)

        dataset[f'log_ffd_high_{param}'] = np.log(dataset['ffdHigh']).shift(param//2)
        dataset[f'log_ffd_low_{param}'] = np.log(dataset['ffdLow']).shift(param//2)
        dataset[f'log_ffd_open_{param}'] = np.log(dataset['ffdOpen']).shift(param//2)
        dataset[f'log_ffd_close_{param}'] = np.log(dataset['ffdClose']).shift(param//2)

        # this feature is kind of redundnant
        #dataset[f'roc_{param}'] = ta.ROC(dataset['close'], timeperiod=10)
        dataset[f'trix_{param}'] = ta.TRIX(dataset['close'], timeperiod=param)
        dataset[f'cmo_{param}'] = ta.CMO(dataset['close'], timeperiod=param)
        dataset[f'bop_smomth_{param}'] = ta.BOP(ta.SMA(dataset['open'], timeperiod=param), 
                                                ta.SMA(dataset['high'], timeperiod=param), 
                                                ta.SMA(dataset['low'], timeperiod=param), 
                                                ta.SMA(dataset['close'], timeperiod=param))
        
        dataset[f'minus_di_{param}'] = ta.MINUS_DI(dataset['high'], dataset['low'], dataset['close'], timeperiod=param)
        dataset[f'aroon_osc_{param}'] = ta.AROONOSC(dataset['high'], dataset['low'], timeperiod=param)
        dataset[f'ult_osc_{param}'] = ta.ULTOSC(dataset['high'], dataset['low'], dataset['close'], timeperiod1=param//3, timeperiod2=param//2, timeperiod3=param)

    # difference between lines needs to be above threshold , 5 or 10
    dataset[f'ht_sine'], dataset[f'ht_leadsine'] = ta.HT_SINE(dataset['close'])

    nonFeatures = ['open', 'high', 'low', 'close', 'volume', 
                    'tick_num', 'num_ticks_sampled','fast_mavg', 'slow_mavg', 'side',
                    'ffdOpen', 'ffdHigh', 'ffdLow', 'ffdClose']
    featureList = [x for x in dataset.columns if x not in nonFeatures]

    # control for lookahead bias
    dataset[featureList] = dataset[featureList].shift(1)
    #secondaryModelData = secondaryModelData.dropna()
    categoricalFeats = []
    continuousFeats = featureList

    return dataset, featureList, categoricalFeats, continuousFeats, nonFeatures

def allFeaturesTrendLayer(dataset, paramDict):
    contFeatures, _, _, _, _ = allFeatures(dataset, paramDict)
    continuousFeats = []
    categoricalFeats = []

    for param in paramDict:
        #dataset[f'sma_{param}_trend'] = np.where(contFeatures[f'sma_{param}'] < dataset['close'], 1, -1)

        #dataset[f'wma_{param}_trend'] = np.where(contFeatures[f'wma_{param}'] < dataset['close'], 1, -1)

        dataset[f'sma_ffd_{param}_trend'] = np.where(contFeatures[f'sma_ffd_{param}'] < dataset['close'], 1, -1)
        dataset[f'wma_ffd_{param}_trend'] = np.where(contFeatures[f'wma_ffd_{param}'] < dataset['close'], 1, -1)
        dataset[f'ema_ffd_{param}_trend'] = np.where(contFeatures[f'ema_ffd_{param}'] < dataset['close'], 1, -1)

        dataset[f'mom_{param}_trend'] = np.where(contFeatures[f'mom_{param}'] > 0, 1, -1)

        cciConditions = [contFeatures[f'CCI_{param}'] > 200, 
                        contFeatures[f'CCI_{param}'] < -200,]
        cciChoices = [-1, 1]
        dataset[f'CCI_{param}_trend'] = np.select(cciConditions, cciChoices)

        dataset[f'ad_osc_{param}_trend'] = np.where(contFeatures[f'ad_osc_{param}'] > contFeatures[f'ad_osc_{param}'].shift(1), 1, -1)
        dataset[f'ad_osc_ffd_{param}_trend'] = np.where(contFeatures[f'ad_osc_ffd_{param}'] > contFeatures[f'ad_osc_ffd_{param}'].shift(1), 1, -1)

        dataset[f'willR_{param}_trend'] = np.where(contFeatures[f'willR_{param}'] > contFeatures[f'willR_{param}'].shift(1), 1, -1)

        # can use macdhist too
        dataset[f'macd_{param}_trend'] = np.where(contFeatures[f'macd_{param}'] > contFeatures[f'macd_{param}'].shift(1), 1, -1)
        macd_hist_conditions = [
            (contFeatures[f'macd_hist_{param}'] > contFeatures[f'macd_hist_{param}'].shift(1)) &
            (contFeatures[f'macd_hist_{param}'] > 0)
        ]
        dataset[f'macd_hist_{param}_trend'] = np.where(macd_hist_conditions[0], 1, -1)

        macd_signal_conditions = [
            (contFeatures[f'macd_{param}'] > contFeatures[f'macd_signal_{param}'] ) &
            (contFeatures[f'macd_{param}'] < 0),
            (contFeatures[f'macd_{param}'] < contFeatures[f'macd_signal_{param}'])  &
            (contFeatures[f'macd_{param}'] > 0),
        ]
        macd_signal_choices = [1, -1]
        dataset[f'macd_signal_{param}_trend'] = np.select(macd_signal_conditions, macd_signal_choices)

        dataset[f'stoch_k_{param}_trend'] = np.where(contFeatures[f'stoch_k_{param}'] > contFeatures[f'stoch_k_{param}'].shift(1), 1, -1)
        dataset[f'stoch_d_{param}_trend'] = np.where(contFeatures[f'stoch_k_{param}'] > contFeatures[f'stoch_d_{param}'].shift(1), 1, -1)

        rsiConditions = [contFeatures[f'rsi_{param}'] > 70,
                        (contFeatures[f'rsi_{param}'] <= 70) & (contFeatures[f'rsi_{param}'] >= 30) & 
                        (contFeatures[f'rsi_{param}'] > contFeatures[f'rsi_{param}'].shift(1)),
                        (contFeatures[f'rsi_{param}'] <= 70) & (contFeatures[f'rsi_{param}'] >= 30) & 
                        (contFeatures[f'rsi_{param}'] < contFeatures[f'rsi_{param}'].shift(1)),
                        contFeatures[f'rsi_{param}'] < 30]
        rsiChoices = [1, 1, -1, -1]
        dataset[f'rsi_{param}_trend'] = np.select(rsiConditions, rsiChoices)


        dataset[f'bbandDiff_{param}'] = contFeatures[f'upperband_ffd_{param}'] -contFeatures[f'lowerband_ffd_{param}']
        dataset[f'bbandUpper_{param}_trend'] = np.where(dataset['ffdClose'] > contFeatures[f'upperband_ffd_{param}'], 1, 0)
        dataset[f'bbandLower_{param}_trend'] = np.where(dataset['ffdClose'] < contFeatures[f'lowerband_ffd_{param}'], -1, 0)


        atr_conditions = [
             contFeatures[f'atr_{param}'] * dataset['close'].shift(param//2) * 1.5  < dataset['close'],
             contFeatures[f'atr_{param}'] * 1.5 * dataset['close'].shift(param//2) > dataset['close'],
        ]
        atr_choices = [
            1, -1
        ]
        dataset[f'atr_{param}_trend'] = np.select(atr_conditions, atr_choices)

        dataset[f'ppo_{param}_trend']  = np.where(contFeatures[f'ppo_{param}'] > 0, 1, -1)

        #dataset[f'dmi_{param}'] = contFeatures[f'dmi_{param}']

        dataset[f'trix_{param}_trend'] = np.where(contFeatures[f'trix_{param}']>0, 1, -1)
        cmoConditions = [contFeatures[f'cmo_{param}'] > 45, 
                        contFeatures[f'cmo_{param}'] < -45,]
        cmoChoices = [-1, 1]
        dataset[f'cmo_{param}_trend'] = np.select(cmoConditions, cmoChoices)
        dataset[f'bop_smomth_{param}_trend'] = np.where( contFeatures[f'bop_smomth_{param}']>0, 1, -1)

        dataset[f'aroon_osc_{param}_trend'] = np.where(contFeatures[f'aroon_osc_{param}'] > 0, 1, -1)
        ult_osc_conditions = [
            contFeatures[f'ult_osc_{param}'] > 50,
            contFeatures[f'ult_osc_{param}'] < 50,
            contFeatures[f'ult_osc_{param}'] > 70,
            contFeatures[f'ult_osc_{param}'] < 30,
        ]
        ult_osc_choices = [1,-1,-1,1]
        dataset[f'ult_osc_{param}_trend'] = np.select(ult_osc_conditions, ult_osc_choices)

        # difference between lines needs to be above threshold , 5 or 10
        ht_sine_conditions = [(contFeatures[f'ht_sine'] - contFeatures[f'ht_leadsine']) > 0.5,
                                (contFeatures[f'ht_sine'] - contFeatures[f'ht_leadsine']) < -0.5]
        ht_sine_choices = [1,-1]
        dataset[f'ht_sine_trend'] = np.select(ht_sine_conditions, ht_sine_choices)
        
    
        dataset[f'log_ffd_high_{param}'] = np.log(dataset['ffdHigh']).shift(param//2)
        dataset[f'log_ffd_low_{param}'] = np.log(dataset['ffdLow']).shift(param//2)
        dataset[f'log_ffd_open_{param}'] = np.log(dataset['ffdOpen']).shift(param//2)
        dataset[f'log_ffd_close_{param}'] = np.log(dataset['ffdClose']).shift(param//2)


        dataset[f'autocorr_{param}'] = contFeatures[f'autocorr_{param}'] 
        dataset[f'autocorr_ffd_{param}'] = contFeatures[f'autocorr_ffd_{param}']
        dataset[f'volatility_ffd_{param}'] = contFeatures[f'volatility_ffd_{param}'] 

        dataset[f'volume_{param}_trend'] = np.where(dataset['volume'].rolling(param//2).mean() > dataset['volume'].rolling(param//2).mean().shift(1), 1, -1)


        continuousFeats.append(f'bbandDiff_{param}', )
        continuousFeats.append(f'autocorr_{param}',)
        continuousFeats.append(f'volatility_ffd_{param}', )
        continuousFeats.append(f'autocorr_ffd_{param}',)
        continuousFeats.append(f'log_ffd_high_{param}',)
        continuousFeats.append(f'log_ffd_low_{param}',)
        continuousFeats.append(f'log_ffd_open_{param}',)
        continuousFeats.append(f'log_ffd_close_{param}',)

    nonFeatures = ['open', 'high', 'low', 'close', 'volume', 
                    'tick_num', 'num_ticks_sampled','fast_mavg', 'slow_mavg', 'side',
                    'ffdOpen', 'ffdHigh', 'ffdLow', 'ffdClose']
    categoricalFeats = [x for x in dataset.columns if (x not in continuousFeats) and (x not in nonFeatures)]
    featureList = continuousFeats + categoricalFeats

    return dataset, featureList, categoricalFeats, continuousFeats, nonFeatures

def trend0(dataset, period, fracdiffClose = None):
    if fracdiffClose is not None:
        returns = fracdiffClose
    else:
        returns = dataset['close'].pct_change()
    
    return returns.rolling(period-1).apply(powerWeightChanges)

def powerWeightChanges(returns):
    power = 1.5
    weights = np.array([])
    for i in range(len(returns)):
        w = i + 1
        weights = np.append(weights, w**power)
    return sum(returns * weights) / sum(weights)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler


# import local packages
from qfml_workflow import dataStructures
from qfml_workflow import cusumFilter
from qfml_workflow import labeling
from qfml_workflow import multiprocess
from qfml_workflow import sampleWeights
from qfml_workflow import fractionalDifferencing
from qfml_workflow import betSizing
from qfml_workflow import crossValidation
from qfml_workflow import featureImportance
from qfml_workflow import backtesting
from qfml_workflow import hyperparamTuning

class SimpleModel():
    def __init__(self, bars, datapath=None):
        # model params
        #TODO: add embargo percents and other hidden params
        self.basic_params = {
                # simple moving average params
                'fast_window' : 20,
                'slow_window' : 50,
                # cusum filter params
                'volatility_span' : 100,
                # labeling params
                'time_expiry' : 1,
                'pt_sl' : [1,2],
                'min_ret' : 0.005,
                # sample weights
                'clfLastW' : 0.5,
                # bet sizing
                'step_size' : 0.05,
                'num_classes' : 10,
                # cpcv backtesting
                'n_splits' : 10,
                'n_folds' : 2,
                }
        self.hyper_params = {
                'max_depth':[2, 3, 4, 5, 7],
                'n_estimators':[1, 10, 25, 50, 100, 256, 512],
                'random_state':[12]
                }
        self.dBars = bars
        self.X = None
        self.y = None

    def getData(self):
        dbars_datapath = 'E:/Prado Research/qfml_workflow clean/qfml_workflow/data/interim/SPY_dollarbars.csv'
        # raw data
        #datapath = 'F:/prado research data/trades_short_condensed/SPY_trades_condensed.csv'

        # Uncomment this line to create dollar bars from raw data file and save to csv
        #dollar_bars = dataStructures.dollar_bars(datapath, 70000000, outfp='E:/Prado Research/qfml_workflow clean/qfml_workflow/data/interim/SPY_dollarbars.csv')
        dollar_bars = pd.read_csv(dbars_datapath)
        
        if dollar_bars['timestamp'].duplicated(keep='first').sum() > 0:
            dollar_bars['timestamp'] = pd.to_datetime(dollar_bars['timestamp'])
            duplicates = dollar_bars[dollar_bars.duplicated(subset='timestamp')]

            for idx, duplicate in duplicates.iterrows():
                # add offset of 1 millisecond
                dollar_bars.at[idx, 'timestamp'] = duplicate['timestamp'] + pd.Timedelta(milliseconds=1) 

            # set timestamp as index
            dollar_bars['timestamp'] = pd.to_datetime(dollar_bars['timestamp'], unit='ns')
            dollar_bars.set_index('timestamp', drop=True, inplace=True)

        self.dBars = dollar_bars

        return dollar_bars.info
    
    def getDailyVolatility(self, data=None):
        if data is None:
            return cusumFilter.getDailyVol(self.dBars['close'], span0=100)
        else:
            return cusumFilter.getDailyVol(data['close'], span0=100)
    
    def createBenchmarkModel(self):
        fast_window = self.basic_params['fast_window']
        slow_window = self.basic_params['slow_window']

        # use ticks sampled as a feature
        nonFeatures = ['open', 'high', 'low', 'close', 'tick_num', 
                       'fast_mavg', 'slow_mavg','stationaryClose']
                       #'side', 

        ## IDEA: RETURN SIDE IN X TO MULT W MODEL

        modelData = self.dBars.copy()

        modelData['fast_mavg'] = modelData['close'].rolling(window=fast_window, min_periods=fast_window, center=False).mean()
        modelData['slow_mavg'] = modelData['close'].rolling(window=slow_window, min_periods=slow_window, center=False).mean()

        # Compute sides
        modelData['side'] = np.nan

        long_signals = modelData['fast_mavg'] >= modelData['slow_mavg'] 
        short_signals = modelData['fast_mavg'] < modelData['slow_mavg'] 
        modelData.loc[long_signals, 'side'] = 1
        modelData.loc[short_signals, 'side'] = -1

        

        # lag the signal to remove look-ahead bias
        modelData['side'] = modelData['side'].shift(1)

        # pass just close
        modelData['stationaryClose'] = fractionalDifferencing.getMinFFD_features(np.log(modelData[['close']]))

        #modelData['log_ret'] = np.log(modelData['stationaryClose']).diff()
        modelData['log_ret'] = modelData['stationaryClose'].diff()

        # Momentum
        modelData['mom1'] = modelData['stationaryClose'].pct_change(periods=1)
        modelData['mom2'] = modelData['stationaryClose'].pct_change(periods=2)
        modelData['mom3'] = modelData['stationaryClose'].pct_change(periods=3)
        modelData['mom4'] = modelData['stationaryClose'].pct_change(periods=4)
        modelData['mom5'] = modelData['stationaryClose'].pct_change(periods=5)

        # Volatility
        modelData['volatility_50'] = modelData['log_ret'].rolling(window=50, min_periods=50, center=False).std()
        modelData['volatility_25'] = modelData['log_ret'].rolling(window=25, min_periods=25, center=False).std()
        modelData['volatility_10'] = modelData['log_ret'].rolling(window=10, min_periods=10, center=False).std()

        # Get the various log -t returns
        modelData['log_t1'] = modelData['log_ret'].shift(1)
        modelData['log_t2'] = modelData['log_ret'].shift(2)
        modelData['log_t3'] = modelData['log_ret'].shift(3)
        modelData['log_t4'] = modelData['log_ret'].shift(4)
        modelData['log_t5'] = modelData['log_ret'].shift(5)

        
        # remove look-ahead
        featureNames = ['log_ret', 'mom1', 'mom2', 'mom3', 'mom4', 'mom5', 
                        'volatility_10', 'volatility_25', 'volatility_50', 'fast_mavg', 'slow_mavg',
                         'log_t1', 'log_t2', 'log_t3', 'log_t4', 'log_t5']
        # SHIFT TO DEAL WITH LOOKAHEAD
        modelData[featureNames] = modelData[featureNames].shift(1)

        modelData.dropna(axis=0, how='any', inplace=True)


        # get cusum events #########
        dailyVolatility = self.getDailyVolatility(modelData)
        cusumEvents = cusumFilter.getTEvents(modelData['close'],h=dailyVolatility.mean())

        # tripple barrier labels
        pt_sl = self.basic_params['pt_sl']
        min_ret = self.basic_params['min_ret']

        vertical_barriers = labeling.addVerticalBarrier(cusumEvents, modelData['close'], numDays=7)
        dailyVolatility = dailyVolatility.reindex(cusumEvents)

#########TODO: check side and bin with meta labeling
        ## TODO: PROBABLY RIGHT read if including side removes the use of the tripple barrier,
        #           see how the lower barriers are used. actually
        #          it doesnt matter which barrier is touched, because they are all the same
        #         same signal right? signal to sell or stop. ****
        # don't include sides because this isnt meta-labeling
        tripleBarrierEvents = labeling.getEvents(close=modelData['close'],
                                               tEvents=cusumEvents,
                                               ptSl=pt_sl,
                                               trgt=dailyVolatility,
                                               minRet=min_ret,
                                               numThreads=3,
                                               t1=vertical_barriers, # vertical barrier times
                                               side=None)
                                               #side=modelData['side']) # side prediction
        # in metha implimentation, returns have side prediction built in
        labels = labeling.getBins(tripleBarrierEvents, modelData['close'])

        print("done labeling")
                # Follow fracdiff recommended implementation in fracDiff chapter
        features = modelData.drop(nonFeatures, axis=1)


        # TODO: should the entire model be cusumed, or should the cusum be just for the
        # features that need fractional differencing?

        features = features.cumsum()

        ##| TODO: np.log features before using fracdiff? LOG DOESNT WORK WITH NEGATIVES
        nstFeats = fractionalDifferencing.getNonStationaryFeats(features)

        print(nstFeats)

        #modelData = fractionalDifferencing.getMinFFD_features(data=modelData[nstFeats])


        return modelData, labels['bin'], tripleBarrierEvents

   
    
    def getSampleWeights(self, X_train, y_train, events_train):
        weights = sampleWeights.getWeights(X_train, events_train)
        returnBasedWeights = weights['w']
        # clfLastWeight is end value for decay
        timeBasedWeightsLinearDecay = sampleWeights.getTimeDecay(weights['tW'], clfLastW=self.hyper_params['clfLastW'])
        
        return timeBasedWeightsLinearDecay, returnBasedWeights
    
    # TODO: Determine if all features should be cusumed
    def applyFractionalDiff(self, data):
        data = data.cusum()
        stationaryFeatures = []

        for feature in data.columns:
            feat_adf=adfuller(data[feature],maxlag=1,regression='c',autolag=None)
            if feat_adf[1] < 0.05:
                stationaryFeatures.append(feature)

        minFFd_feats = fractionalDifferencing.getMinFFD_features(data[~stationaryFeatures])
        data[~stationaryFeatures] = minFFd_feats[~stationaryFeatures]

        return data
    
    # TODO: SCALER FROM TEST SHOULD BE USED FOR SCALER FOR TRAIN
    def scaleData(self, X_train, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test

    # TODO: Implement other params here
    def plotFeatureImportance(self, X_train, events_train, method='SFI'):
        X_train = X_train.loc[X_train.index >= events_train.index[0]]
        events_train = events_train.loc[events_train.index >= X_train.index[0]]

        featImp = featureImportance.featImportance(X_train, events_train)
        featureImportance.plotFeatImportance('', featImp[0], featImp[1], featImp[2], method='SFI')

    def getBetSizing(self, metaModel, X_test, y_pred, events_test):
        # step size for discretization
        stepSize = self.hyper_params['step_size']
        probability = np.max(metaModel.predict_proba(X_test), axis=1)
        prediction = pd.Series(y_pred, index=X_test.index)
        numClasses = self.hyper_params['num_classes']
        numThreads = 1
        #testEvents = events_test.loc[events_test.index >= X_test.index[0]]
        sizes = betSizing.getSignal(events_test, stepSize, probability, prediction, numClasses, numThreads)

        return sizes

    def plotModel(self, model, X_train, y_train, X_test, y_test):
        print('TRAINING###################')
        y_pred_rf = model.predict_proba(X_train)[:, 1]
        y_pred = model.predict(X_train)
        fpr_rf, tpr_rf, _ = roc_curve(y_train, y_pred_rf)
        print(classification_report(y_train, y_pred))

        print("Confusion Matrix")
        print(confusion_matrix(y_train, y_pred))

        print('')
        print("Accuracy")
        print(accuracy_score(y_train, y_pred))

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_rf, tpr_rf, label='RF')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()
        print('##############')
        print('TESTING###################')
        y_pred_rf = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix")
        print(confusion_matrix(y_test, y_pred))

        print('')
        print("Accuracy")
        print(accuracy_score(y_test, y_pred))

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_rf, tpr_rf, label='RF')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()

    # TODO: Should embargo percent be the same across all functions, or different?
    def runCombinatorialPurgedKFoldBacktest(self, X, y, n_splits, n_test_splits, ):
        n_splits = 10
        n_test_splits = 2
        embargo_td = backtesting.embargoFromPercent(0, X_train_feats.index, 0.01)

        prediction_matrix = np.matrix((GROUPS, SPLITS))

        # testSets is a list of tuples for the test sets in each split.
        testSets = []

        # generate df with train and test indexes
        for train_ind, test_ind in cv_gen.split(X.reindex(x_df.index), x_df['y'], x_df['pred_times'], x_df['eval_times']):
        #train model
        # test model
            X_train = X.iloc[train_ind]
            X_test = X.iloc[test_ind]
            y_train = y.iloc[train_ind]
            y_test = y.iloc[test_ind]

            # model should be a class
           # predictions = self.createMetaModel(X_train, y_train, X_test, y_test)

            # get cusum events #########
            dailyVolatility = self.getDailyVolatility(X_train)
            cusumEvents = cusumFilter.getTEvents(X_train['close'],h=dailyVolatility.mean())

            # tripple barrier labels
            pt_sl = self.basic_params['pt_sl']
            min_ret = self.basic_params['min_ret']

            vertical_barriers = labeling.addVerticalBarrier(cusumEvents, X_train['close'], numDays=1)
            dailyVolatility = dailyVolatility.reindex(cusumEvents)

    #########TODO: check side and bin with meta labeling
            # don't include sides because this isnt meta-labeling
            tripleBarrierEvents = labeling.getEvents(close=X_train['close'],
                                                tEvents=cusumEvents,
                                                ptSl=pt_sl,
                                                trgt=dailyVolatility,
                                                minRet=min_ret,
                                                numThreads=3,
                                                t1=vertical_barriers,) # vertical barrier times
                                                #side=X_train['side']) # side prediction
            rf = RandomForestClassifier(criterion='entropy', class_weight='balanced', random_state=self.hyper_params.random_state[0])

            pipeline = hyperparamTuning.clfHyperFit(X_train, y_train, tripleBarrierEvents['t1'], rf, self.hyper_params,bagging=1, pctEmbargo=0.01)
            n_estimators, max_depth = pipeline.best_params_['n_estimators'], pipeline.best_params_['max_depth']
        
            #rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,
            #    criterion='entropy', random_state=self.hyper_params.random_state[0], class_weight='balanced')

            rf.set_params(max_depth=max_depth, n_estimators=n_estimators)

            # make predictions
            rf.fit(X_train, y_train.values.ravel())

            probs = np.max(rf.predict_proba(X_test), axis=1)

            return rf.predict(X_test) * probs


    def makePredictions(self, X_train, y_train, X_test, model):

            # model should be a class
           # predictions = self.createMetaModel(X_train, y_train, X_test, y_test)
            '''
            # get cusum events #########
            dailyVolatility = self.getDailyVolatility(X_train)
            cusumEvents = cusumFilter.getTEvents(X_train['close'],h=dailyVolatility.mean())

            # tripple barrier labels
            pt_sl = self.basic_params['pt_sl']
            min_ret = self.basic_params['min_ret']

            vertical_barriers = labeling.addVerticalBarrier(cusumEvents, X_train['close'], numDays=1)
            dailyVolatility = dailyVolatility.reindex(cusumEvents)

    #########TODO: check side and bin with meta labeling
            # don't include sides because this isnt meta-labeling
            tripleBarrierEvents = labeling.getEvents(close=X_train['close'],
                                                tEvents=cusumEvents,
                                                ptSl=pt_sl,
                                                trgt=dailyVolatility,
                                                minRet=min_ret,
                                                numThreads=3,
                                                t1=vertical_barriers,) # vertical barrier times
                                                #side=X_train['side']) # side prediction

            '''

            side_prediction = X_test['side']
            X_train = X_train.drop('side', axis=1)
            X_test = X_test.drop('side', axis=1)

            # Define the scaler object
            trainScaler = MinMaxScaler()

            X_train_scaled = trainScaler.fit_transform(X_train.to_numpy())
            # use the training scaler on the test set so there is no data leakeage
            X_test_scaled = trainScaler.transform(X_test.to_numpy())
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

            # PROBABLY NEEDS TO BE IMPLMENTED HERE BECAUSE TEVENT RET MIGHT
            # OVERLAP WITH NEXT TEST SET AND TRAIN SET
            # get labels at event dates
            X_train = X_train.loc[y_train.index, :]
            #tbEvents = tbEvents.loc[X_train.index, :]
            classifierModel = model

            #rf = RandomForestClassifier(criterion='entropy', class_weight='balanced', random_state=self.hyper_params['random_state'][0])

            #TODO: tbEvents should probably be retrained here on only the training sets
           # trainEvents = tbEvents.reindex(X_train.index).dropna()
            #pipeline = hyperparamTuning.clfHyperFit(X_train, y_train, trainEvents['t1'], rf, self.hyper_params,bagging=1, pctEmbargo=0.01)
            #n_estimators, max_depth = pipeline.best_params_['n_estimators'], pipeline.best_params_['max_depth']
            #n_estimators, max_depth = 4, 25
        
            #rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,
            #    criterion='entropy', random_state=self.hyper_params.random_state[0], class_weight='balanced')

            #rf.set_params(max_depth=max_depth, n_estimators=n_estimators)
            


            # make predictions
            classifierModel.fit(X_train, y_train.values.ravel())

            probs = np.max(classifierModel.predict_proba(X_test), axis=1)

            #return rf.predict(X.test) * tbEvents['ret']
            #TODO: need to incorporate side
            return pd.Series(classifierModel.predict(X_test) * probs * side_prediction, index=X_test.index) 

            '''
            CUSUM events use daily volatility,
             so the labels are not as frequent. if using time bars,
             it might make more sense to lower the volatility frequency

            should label * signal be returned?
            
            '''


    def makePredictionsBetSizing(self, X_train, y_train, X_test, model, tbEvents, stepSize=0.05, numClasses=2):

            # model should be a class
           # predictions = self.createMetaModel(X_train, y_train, X_test, y_test)
            '''
            # get cusum events #########
            dailyVolatility = self.getDailyVolatility(X_train)
            cusumEvents = cusumFilter.getTEvents(X_train['close'],h=dailyVolatility.mean())

            # tripple barrier labels
            pt_sl = self.basic_params['pt_sl']
            min_ret = self.basic_params['min_ret']

            vertical_barriers = labeling.addVerticalBarrier(cusumEvents, X_train['close'], numDays=1)
            dailyVolatility = dailyVolatility.reindex(cusumEvents)

    #########TODO: check side and bin with meta labeling
            # don't include sides because this isnt meta-labeling
            tripleBarrierEvents = labeling.getEvents(close=X_train['close'],
                                                tEvents=cusumEvents,
                                                ptSl=pt_sl,
                                                trgt=dailyVolatility,
                                                minRet=min_ret,
                                                numThreads=3,
                                                t1=vertical_barriers,) # vertical barrier times
                                                #side=X_train['side']) # side prediction

            '''

            side_prediction = X_test['side']
            X_train = X_train.drop('side', axis=1)
            X_test = X_test.drop('side', axis=1)

            # Define the scaler object
            trainScaler = MinMaxScaler()

            X_train_scaled = trainScaler.fit_transform(X_train.to_numpy())
            # use the training scaler on the test set so there is no data leakeage
            X_test_scaled = trainScaler.transform(X_test.to_numpy())
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

            # PROBABLY NEEDS TO BE IMPLMENTED HERE BECAUSE TEVENT RET MIGHT
            # OVERLAP WITH NEXT TEST SET AND TRAIN SET
            # get labels at event dates
            X_train = X_train.loc[y_train.index, :]
            #tbEvents = tbEvents.loc[X_train.index, :]
            classifierModel = model

            #rf = RandomForestClassifier(criterion='entropy', class_weight='balanced', random_state=self.hyper_params['random_state'][0])

            #TODO: tbEvents should probably be retrained here on only the training sets
           # trainEvents = tbEvents.reindex(X_train.index).dropna()
            #pipeline = hyperparamTuning.clfHyperFit(X_train, y_train, trainEvents['t1'], rf, self.hyper_params,bagging=1, pctEmbargo=0.01)
            #n_estimators, max_depth = pipeline.best_params_['n_estimators'], pipeline.best_params_['max_depth']
            #n_estimators, max_depth = 4, 25
        
            #rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,
            #    criterion='entropy', random_state=self.hyper_params.random_state[0], class_weight='balanced')

            #rf.set_params(max_depth=max_depth, n_estimators=n_estimators)
            


            # make predictions
            classifierModel.fit(X_train, y_train.values.ravel())

            probs = np.max(classifierModel.predict_proba(X_test), axis=1)

            prediction = classifierModel.predict(X_test)

            betSizes = betSizing.getSignal(tbEvents,stepSize,probs,prediction,numClasses,numThreads=1)

            #return rf.predict(X.test) * tbEvents['ret']
            #TODO: need to incorporate side
            #return pd.Series(classifierModel.predict(X_test) * probs * side_prediction, index=X_test.index) 
            return pd.Series(betSizes * side_prediction, index=X_test.index)
    

    def makePredictionsFixed(self, X_train, y_train, X_test, model):

            # model should be a class
           # predictions = self.createMetaModel(X_train, y_train, X_test, y_test)
            '''
            # get cusum events #########
            dailyVolatility = self.getDailyVolatility(X_train)
            cusumEvents = cusumFilter.getTEvents(X_train['close'],h=dailyVolatility.mean())

            # tripple barrier labels
            pt_sl = self.basic_params['pt_sl']
            min_ret = self.basic_params['min_ret']

            vertical_barriers = labeling.addVerticalBarrier(cusumEvents, X_train['close'], numDays=1)
            dailyVolatility = dailyVolatility.reindex(cusumEvents)

    #########TODO: check side and bin with meta labeling
            # don't include sides because this isnt meta-labeling
            tripleBarrierEvents = labeling.getEvents(close=X_train['close'],
                                                tEvents=cusumEvents,
                                                ptSl=pt_sl,
                                                trgt=dailyVolatility,
                                                minRet=min_ret,
                                                numThreads=3,
                                                t1=vertical_barriers,) # vertical barrier times
                                                #side=X_train['side']) # side prediction

            '''

            #side_prediction = X_test['side']
            #X_train = X_train.drop('side', axis=1)
            #X_test = X_test.drop('side', axis=1)

            # Define the scaler object
            #trainScaler = MinMaxScaler()

            #X_train_scaled = trainScaler.fit_transform(X_train.to_numpy())
            # use the training scaler on the test set so there is no data leakeage
            #X_test_scaled = trainScaler.transform(X_test.to_numpy())
            #X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            #X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

            # PROBABLY NEEDS TO BE IMPLMENTED HERE BECAUSE TEVENT RET MIGHT
            # OVERLAP WITH NEXT TEST SET AND TRAIN SET
            # get labels at event dates
            #X_train = X_train.loc[y_train.index, :]
            #tbEvents = tbEvents.loc[X_train.index, :]
            classifierModel = model

            # make predictions
           # classifierModel.fit(X_train, y_train.values.ravel())
            classifierModel.fit(X_train, y_train)

            probs = np.amax(classifierModel.predict_proba(X_test), axis=1)

            #return rf.predict(X.test) * tbEvents['ret']
            #TODO: need to incorporate side
            #return pd.Series(classifierModel.predict(X_test) * probs * side_prediction, index=X_test.index) 
            return pd.Series(classifierModel.predict(X_test) * probs * side_prediction, index=X_test.index) 

            '''
            CUSUM events use daily volatility,
             so the labels are not as frequent. if using time bars,
             it might make more sense to lower the volatility frequency

            should label * signal be returned?
            
            '''
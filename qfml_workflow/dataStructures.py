import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

from tqdm import tqdm
#from src.data_structures import DataStructures as dv1



def time_bars( infp, resolution, chunksize=1000000, outfp=None, numRows=None):
    # for accuracy, last day needs to be 
    if outfp is not None:
        _get_time_bars(infp, resolution, 'dollar')
        print("data saved to csv.")
        return None
    else:
        return _get_bars(infp, resolution, 'dollar', chunksize, outfp, numRows)

def tick_bars( infp, sample_thresh, chunksize=1000000, outfp=None, numRows=None):
    if outfp is not None:
        _get_bars(infp, sample_thresh, 'tick', chunksize, outfp, numRows)
        print("data saved to csv.")
        return None
    else:
        return _get_bars(infp, sample_thresh, 'tick', chunksize, outfp, numRows)

def volume_bars( infp, sample_thresh, chunksize=1000000, outfp=None, numRows=None):
    if outfp is not None:
        _get_bars(infp, sample_thresh, 'volume', chunksize, outfp, numRows)
        print("data saved to csv.")
        return None
    else:
        return _get_bars(infp, sample_thresh, 'volume', chunksize, outfp, numRows)

def dollar_bars( infp, sample_thresh, chunksize=1000000, outfp=None, numRows=None):
    if outfp is not None:
        _get_bars(infp, sample_thresh, 'dollar', chunksize, outfp, numRows)
        #_get_bars_fixed(infp, sample_thresh, 'dollar', chunksize, outfp )
        print("data saved to csv.")
        return None
    else:
        return _get_bars(infp, sample_thresh, 'dollar', chunksize, outfp, numRows)
    
def dollar_bars_dynamic( infp, sample_thresh, chunksize=1000000, outfp=None, numRows=None):
    if outfp is not None:
        _get_bars_dynamic(infp, sample_thresh, 'dollar', chunksize, outfp, numRows)
        #_get_bars_fixed(infp, sample_thresh, 'dollar', chunksize, outfp )
        print("data saved to csv.")
        return None
    else:
        return _get_bars_dynamic(infp, sample_thresh, 'dollar', chunksize, outfp, numRows)

def dollar_bars_dynamic_no_lookahead( infp, sample_thresh, chunksize=1000000, outfp=None, numRows=None):
    if outfp is not None:
        _get_bars_dynamic_no_lookahead(infp, sample_thresh, 'dollar', chunksize, outfp, numRows)
        #_get_bars_fixed(infp, sample_thresh, 'dollar', chunksize, outfp )
        print("data saved to csv.")
        return None
    else:
        return _get_bars_dynamic_no_lookahead(infp, sample_thresh, 'dollar', chunksize, outfp, numRows)

def dollar_bars_dynamic_no_lookahead_second( infp, sample_thresh, chunksize=1000000, outfp=None, numRows=None):
    if outfp is not None:
        _get_bars_dynamic_second_data(infp, sample_thresh, 'dollar', chunksize, outfp, numRows)
        #_get_bars_fixed(infp, sample_thresh, 'dollar', chunksize, outfp )
        print("data saved to csv.")
        return None
    else:
        return _get_bars_dynamic_second_data(infp, sample_thresh, 'dollar', chunksize, outfp, numRows)

def dollar_bars_dynamic_no_lookahead_time_fix( infp, sample_thresh, chunksize=1000000, outfp=None, numRows=None):
    if outfp is not None:
        _get_bars_dynamic_no_lookahead_time_resample(infp, sample_thresh, 'dollar', chunksize, outfp, numRows)
        #_get_bars_fixed(infp, sample_thresh, 'dollar', chunksize, outfp )
        print("data saved to csv.")
        return None
    else:
        return _get_bars_dynamic_no_lookahead_time_resample(infp, sample_thresh, 'dollar', chunksize, outfp, numRows)

def dollar_bars_dynamic_no_lookahead_dollar_market( infp, sample_thresh, chunksize=1000000, outfp=None, numRows=None):
    if outfp is not None:
        _get_bars_dynamic_no_lookahead_sample_during_market(infp, sample_thresh, 'dollar', chunksize, outfp, numRows)
        #_get_bars_fixed(infp, sample_thresh, 'dollar', chunksize, outfp )
        print("data saved to csv.")
        return None
    else:
        return _get_bars_dynamic_no_lookahead_sample_during_market(infp, sample_thresh, 'dollar', chunksize, outfp, numRows)


def _get_bars_fixed(fp, threshold, bar_type, chunksize=1000000, outfp=None):
    cols = ["timestamp", "open", "high", "low", "close", "volume", "tick_num", "num_ticks_sampled"]
    runningHigh = 0
    runningLow = 1E10
    runningThresh = 0
    runningVolume = 0
    runningTickNum = 0
    runningOpen = 0

    chunkNum = 0
    for chunk in tqdm(pd.read_csv(fp, chunksize=chunksize), desc='chunk #'):
        chunkNum+= 1
        chunk.rename(columns={"sip_timestamp": "timestamp", "price": "price", "size": "volume"}, inplace=True)
        outData = []

        for idx, row in chunk.iterrows():
            if bar_type == 'dollar':
                runningThresh += row['volume'] * row['price']
            elif bar_type == 'volume':
                runningThresh += row['volume']
            elif bar_type ==  'tick':
                return ValueError("Not implemented")

            runningVolume += row['volume']
            runningTickNum += 1

            if row['price'] > runningHigh:
                runningHigh = row['price']

            if row['price'] < runningLow:
                runningLow = row['price']

            if runningOpen == 0:
                runningOpen = row['price']



            if runningThresh >= threshold:
                barTs = row['timestamp']
                barClose = row['price']
                barOpen = runningOpen
                barVol = runningVolume
                barHigh = runningHigh
                barLow = runningLow
                barTicksSampled = runningTickNum
                barTickNum = idx
                

                row = {'timestamp': barTs,'open':barOpen, 'high':barHigh, 'low':barLow, 'close':barClose, 'volume':barVol, 'tick_num': barTickNum, 'num_ticks_sampled': barTicksSampled}
                outData.append(row)

                # reset vals
                runningHigh = 0
                runningLow = 1E10
                runningVolume = 0
                runningTickNum = 0
                runningOpen = 0
                runningThresh = 0
        
        if outfp is not None:
            if chunkNum == 1:
                pd.DataFrame(outData).to_csv(outfp, index=False, header=True)
            else:
                pd.DataFrame(outData).to_csv(outfp, mode='a', index=False, header=False)


def _get_bars( fp, factor, bar_type, chunksize=1000000, outfp=None, rowModifier=None):
    cols = ["timestamp", "open", "high", "low", "close", "volume", "tick_num", "num_ticks_sampled"]
    overlap = pd.DataFrame(columns=['timestamp', 'price', 'volume'])
    chunknum = 0
    outdata = np.zeros(shape=(0, 8))
    sample_count = 0
    rolling_val = 0
    open_tick = 0
    numpyOpenTick = 0
    num_rows = 0
    #last_sample_tick = 0
    loadRows = rowModifier
    if rowModifier == None:
        loadRows = sys.maxsize
    
    outframe = pd.DataFrame(columns=cols)
    elapsedBars = 0
    sample_thresh = factor

    for chunk in tqdm(pd.read_csv(fp, chunksize=chunksize, nrows=loadRows), desc='chunk #'):
        chunknum += 1
        chunk.rename(columns={"sip_timestamp": "timestamp", "price": "price", "size": "volume"}, inplace=True)

        chunk = pd.concat([overlap, chunk], axis=0)

        # if converting to datetime use np not pandas?
        #ts = pd.to_datetime(chunk['timestamp'], unit='ns').to_numpy() # might need to convert to datetime first
        ts = chunk['timestamp'].to_numpy()
        price = chunk['price'].to_numpy()
        vol = chunk['volume'].to_numpy()
        approx_dv = np.dot(price, vol)
        # could also try using a 2 month delayed market cap
        if bar_type == 'dollar' and factor < 100:
            dynamicThresh = approx_dv / ((ts[-1] - ts[0]) / 8.64E13)
            dynamicThresh = dynamicThresh * factor
            sample_thresh = dynamicThresh
        #approx_dv = (chunk['price'] * chunk['volume']).sum()
        # still wrong, need to include sample count
        #num_rows = len(outdata) + math.ceil(approx_dv/sample_thresh)
        
        #outdata.resize((num_rows, 8), refcheck=False)
        
        outdata = {key : [] for key in cols}
        chunkSamples = 0

        offsetSplit = chunk.index.values[0]
        numpyOpenTick = open_tick - offsetSplit
        # create bars for chunk
        for i in chunk.index:
            if chunknum == 3:
                #print(i)
                xsf = 0
            #numpyIndex = np.where(chunk.index.values == i)
            pandasIndex = i
            numpyIndex = i - offsetSplit
            
            #numpyOpenTick = np.where(chunk.index.values == open_tick)

            if bar_type == 'dollar':
                rolling_val += vol[numpyIndex] * price[numpyIndex]
            elif bar_type == 'volume':
                rolling_val += vol[numpyIndex]
            elif bar_type == 'tick':
                # could be issue here, tick bars have issue
                rolling_val = numpyIndex - (sample_count * sample_thresh)

            # maybe want to use np.append instead of constantly resizing array
            # if i >= sample_vol create bar:
            if rolling_val >= sample_thresh:
                outdata[cols[0]].append(ts[numpyIndex] )                     # time
                outdata[cols[1]].append(price[numpyOpenTick] )   
                #print("openTick", open_tick, "numpyOpen", numpyOpenTick, "offset",offsetSplit)
                #print(numpyIndex)
                #print(len(chunk))           # open
                outdata[cols[2]].append(np.max(price[numpyOpenTick:numpyIndex+1]))     # high
                outdata[cols[3]].append(np.min(price[numpyOpenTick:numpyIndex+1]))     # low
                outdata[cols[4]].append(price[numpyIndex])                       # close
                outdata[cols[5]].append(np.sum(vol[numpyOpenTick:numpyIndex+1]))    # volume
                outdata[cols[6]].append(i)                            #tick_num
                outdata[cols[7]].append( i+1  - open_tick )              # num_ticks_sampled
                sample_count += 1
                chunkSamples += 1
                # reset volume and define next bar open tick index
                open_tick = i+1
                numpyOpenTick = open_tick - offsetSplit
                rolling_val = 0
                #last_sample_tick = i  
                #row = [ts[i], price[open_tick],  np.max(price[open_tick:i+1]), np.min(price[open_tick:i+1]),
                 #      price[i], np.sum(vol[open_tick:i+1]), i , i+1  - open_tick   ]

                #outdata.append(row)
        # no issue, iloc goes specifically by row number. use loc for exact index
        
        overlap = chunk.iloc[numpyOpenTick:] 
        #reset open tick
        #open_tick = 0    
        rolling_val = 0 

        if outfp is not None:
            if chunknum == 1:
                pd.DataFrame.from_dict(outdata).to_csv(outfp, index=False, header=True)
            else:
               pd.DataFrame.from_dict(outdata).to_csv(outfp, mode='a', index=False, header=False)
            #outdata = np.empty((num_rows, 8))
            #num_rows = 0    
            #sample_count = 0
        else:
            outframe = pd.concat([outframe, pd.DataFrame.from_dict(outdata)], ignore_index=True)

    '''    
    if outfp is not None:
        processedObject = pd.read_csv(outfp)
        processedObject = processedObject.sort_values('timestamp')
        processedObject.to_csv(outfp, index=False)
    '''

    # might need to add [:sample_count]
    return None if (outfp is not None) else outframe.sort_values('timestamp')

# need to implement
def _get_time_bars():
    pass


def _get_bars_dynamic( fp, factor, bar_type, chunksize=1000000, outfp=None, rowModifier=None):
    cols = ["timestamp", "open", "high", "low", "close", "volume", "tick_num", "num_ticks_sampled"]
    overlap = pd.DataFrame(columns=['timestamp', 'price', 'volume'])
    chunknum = 0
    outdata = np.zeros(shape=(0, 8))
    sample_count = 0
    rolling_val = 0
    open_tick = 0
    numpyOpenTick = 0
    num_rows = 0
    #last_sample_tick = 0
    loadRows = rowModifier
    if rowModifier == None:
        loadRows = sys.maxsize
    
    outframe = pd.DataFrame(columns=cols)
    elapsedBars = 0
    sample_thresh = factor

    for chunk in tqdm(pd.read_csv(fp, chunksize=chunksize, nrows=loadRows), desc='chunk #'):
        chunknum += 1
        chunk.rename(columns={"sip_timestamp": "timestamp", "price": "price", "size": "volume"}, inplace=True)

        chunk = pd.concat([overlap, chunk], axis=0)

        # if converting to datetime use np not pandas?
        #ts = pd.to_datetime(chunk['timestamp'], unit='ns').to_numpy() # might need to convert to datetime first
        ts = chunk['timestamp'].to_numpy()
        price = chunk['price'].to_numpy()
        vol = chunk['volume'].to_numpy()
        approx_dv = np.dot(price, vol)
        # could also try using a 2 month delayed market cap
        if bar_type == 'dollar':# and factor < 100:
            #dynamicThresh = approx_dv / ((ts[-1] - ts[0]) / 8.64E13)
            #dynamicThresh = dynamicThresh * factor
            timePeriod = (pd.to_datetime(chunk['timestamp'].iloc[-1]) - pd.to_datetime(chunk['timestamp'].iloc[0])).days
            rollPeriod = math.ceil((1/(timePeriod+1)) * 50000)
            volatilityEst = chunk['price'].pct_change().rolling(rollPeriod).std().fillna(method='backfill')
            
        #approx_dv = (chunk['price'] * chunk['volume']).sum()
        # still wrong, need to include sample count
        #num_rows = len(outdata) + math.ceil(approx_dv/sample_thresh)
        
        #outdata.resize((num_rows, 8), refcheck=False)
        
        outdata = {key : [] for key in cols}
        chunkSamples = 0

        offsetSplit = chunk.index.values[0]
        numpyOpenTick = open_tick - offsetSplit
        # create bars for chunk
        for i in chunk.index:
            if chunknum == 3:
                #print(i)
                xsf = 0
            #numpyIndex = np.where(chunk.index.values == i)
            pandasIndex = i
            numpyIndex = i - offsetSplit

            
            dynamicThresh = (approx_dv/timePeriod) *(0.1*(1/(1-abs(volatilityEst.loc[i]*100))))
            sample_thresh = dynamicThresh
            
            #numpyOpenTick = np.where(chunk.index.values == open_tick)

            if bar_type == 'dollar':
                rolling_val += vol[numpyIndex] * price[numpyIndex]
            elif bar_type == 'volume':
                rolling_val += vol[numpyIndex]
            elif bar_type == 'tick':
                # could be issue here, tick bars have issue
                rolling_val = numpyIndex - (sample_count * sample_thresh)

            # maybe want to use np.append instead of constantly resizing array
            # if i >= sample_vol create bar:
            if rolling_val >= sample_thresh:
                outdata[cols[0]].append(ts[numpyIndex] )                     # time
                outdata[cols[1]].append(price[numpyOpenTick] )   
                #print("openTick", open_tick, "numpyOpen", numpyOpenTick, "offset",offsetSplit)
                #print(numpyIndex)
                #print(len(chunk))           # open
                outdata[cols[2]].append(np.max(price[numpyOpenTick:numpyIndex+1]))     # high
                outdata[cols[3]].append(np.min(price[numpyOpenTick:numpyIndex+1]))     # low
                outdata[cols[4]].append(price[numpyIndex])                       # close
                outdata[cols[5]].append(np.sum(vol[numpyOpenTick:numpyIndex+1]))    # volume
                outdata[cols[6]].append(i)                            #tick_num
                outdata[cols[7]].append( i+1  - open_tick )              # num_ticks_sampled
                sample_count += 1
                chunkSamples += 1
                # reset volume and define next bar open tick index
                open_tick = i+1
                numpyOpenTick = open_tick - offsetSplit
                rolling_val = 0
                #last_sample_tick = i  
                #row = [ts[i], price[open_tick],  np.max(price[open_tick:i+1]), np.min(price[open_tick:i+1]),
                 #      price[i], np.sum(vol[open_tick:i+1]), i , i+1  - open_tick   ]

                #outdata.append(row)
        # no issue, iloc goes specifically by row number. use loc for exact index
        
        overlap = chunk.iloc[numpyOpenTick:] 
        #reset open tick
        #open_tick = 0    
        rolling_val = 0 

        if outfp is not None:
            if chunknum == 1:
                pd.DataFrame.from_dict(outdata).to_csv(outfp, index=False, header=True)
            else:
               pd.DataFrame.from_dict(outdata).to_csv(outfp, mode='a', index=False, header=False)
            #outdata = np.empty((num_rows, 8))
            #num_rows = 0    
            #sample_count = 0
        else:
            outframe = pd.concat([outframe, pd.DataFrame.from_dict(outdata)], ignore_index=True)

    '''    
    if outfp is not None:
        processedObject = pd.read_csv(outfp)
        processedObject = processedObject.sort_values('timestamp')
        processedObject.to_csv(outfp, index=False)
    '''

    # might need to add [:sample_count]
    return None if (outfp is not None) else outframe.sort_values('timestamp')




def _get_bars_dynamic_no_lookahead( fp, factor, bar_type, chunksize=1000000, outfp=None, rowModifier=None):
    cols = ["timestamp", "open", "high", "low", "close", "volume", "tick_num", "num_ticks_sampled"]
    overlap = pd.DataFrame(columns=['timestamp', 'price', 'volume'])
    #overlap = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    chunknum = 0
    outdata = np.zeros(shape=(0, 8))
    sample_count = 0
    rolling_val = 0
    open_tick = 0
    numpyOpenTick = 0
    num_rows = 0
    #last_sample_tick = 0
    loadRows = rowModifier
    if rowModifier == None:
        loadRows = sys.maxsize
    
    outframe = pd.DataFrame(columns=cols)
    elapsedBars = 0
    sample_thresh = factor

    for chunk in tqdm(pd.read_csv(fp, chunksize=chunksize, nrows=loadRows), desc='chunk #'):
        chunknum += 1
        chunk.rename(columns={"sip_timestamp": "timestamp", "price": "price", "size": "volume"}, inplace=True)
        

        chunk = pd.concat([overlap, chunk], axis=0)

        # if converting to datetime use np not pandas?
        #ts = pd.to_datetime(chunk['timestamp'], unit='ns').to_numpy() # might need to convert to datetime first
        ts = chunk['timestamp'].to_numpy()
        price = chunk['price'].to_numpy()
        vol = chunk['volume'].to_numpy()
        approx_dv = np.dot(price, vol)
        # could also try using a 2 month delayed market cap
        if bar_type == 'dollar':# and factor < 100:
            #dynamicThresh = approx_dv / ((ts[-1] - ts[0]) / 8.64E13)
            #dynamicThresh = dynamicThresh * factor
            timePeriod = (pd.to_datetime(chunk['timestamp'].iloc[-1]) - pd.to_datetime(chunk['timestamp'].iloc[0])).days
            rollPeriod = math.ceil((1/(timePeriod+1)) * 50000)
            volatilityEst = chunk['price'].pct_change().rolling(rollPeriod).std().fillna(method='backfill')
            
        #approx_dv = (chunk['price'] * chunk['volume']).sum()
        # still wrong, need to include sample count
        #num_rows = len(outdata) + math.ceil(approx_dv/sample_thresh)
        
        #outdata.resize((num_rows, 8), refcheck=False)
        
        outdata = {key : [] for key in cols}
        chunkSamples = 0

        offsetSplit = chunk.index.values[0]
        numpyOpenTick = open_tick - offsetSplit
        # create bars for chunk
        for i in chunk.index:
            if chunknum == 3:
                #print(i)
                xsf = 0
            #numpyIndex = np.where(chunk.index.values == i)
            pandasIndex = i
            numpyIndex = i - offsetSplit

            
            dynamicThresh = (approx_dv/timePeriod) *(0.1*(1/(1-abs(volatilityEst.loc[i]*100))))
            #sample_thresh = dynamicThresh
            

            if chunknum ==1:
                sample_thresh = factor
            else:
                sample_thresh = nextThresh

            nextThresh = dynamicThresh

            #numpyOpenTick = np.where(chunk.index.values == open_tick)

            if bar_type == 'dollar':
                rolling_val += vol[numpyIndex] * price[numpyIndex]
            elif bar_type == 'volume':
                rolling_val += vol[numpyIndex]
            elif bar_type == 'tick':
                # could be issue here, tick bars have issue
                rolling_val = numpyIndex - (sample_count * sample_thresh)

            # maybe want to use np.append instead of constantly resizing array
            # if i >= sample_vol create bar:
            if rolling_val >= sample_thresh:
                outdata[cols[0]].append(ts[numpyIndex] )                     # time
                outdata[cols[1]].append(price[numpyOpenTick] )   
                #print("openTick", open_tick, "numpyOpen", numpyOpenTick, "offset",offsetSplit)
                #print(numpyIndex)
                #print(len(chunk))           # open
                outdata[cols[2]].append(np.max(price[numpyOpenTick:numpyIndex+1]))     # high
                outdata[cols[3]].append(np.min(price[numpyOpenTick:numpyIndex+1]))     # low
                outdata[cols[4]].append(price[numpyIndex])                       # close
                outdata[cols[5]].append(np.sum(vol[numpyOpenTick:numpyIndex+1]))    # volume
                outdata[cols[6]].append(i)                            #tick_num
                outdata[cols[7]].append( i+1  - open_tick )              # num_ticks_sampled
                sample_count += 1
                chunkSamples += 1
                # reset volume and define next bar open tick index
                open_tick = i+1
                numpyOpenTick = open_tick - offsetSplit
                rolling_val = 0
                #last_sample_tick = i  
                #row = [ts[i], price[open_tick],  np.max(price[open_tick:i+1]), np.min(price[open_tick:i+1]),
                 #      price[i], np.sum(vol[open_tick:i+1]), i , i+1  - open_tick   ]

                #outdata.append(row)
        # no issue, iloc goes specifically by row number. use loc for exact index
        
        overlap = chunk.iloc[numpyOpenTick:] 
        #reset open tick
        #open_tick = 0    
        rolling_val = 0 

        if outfp is not None:
            if chunknum == 1:
                #pd.DataFrame.from_dict(outdata).to_csv(outfp, index=False, header=True)
                #pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).to_csv(outfp, index=False, header=True)
                pd.DataFrame(columns=cols).to_csv(outfp, index=False, header=True)
            else:
                pd.DataFrame.from_dict(outdata).to_csv(outfp, mode='a', index=False, header=False)
            #outdata = np.empty((num_rows, 8))
            #num_rows = 0    
            #sample_count = 0
        else:
            outframe = pd.concat([outframe, pd.DataFrame.from_dict(outdata)], ignore_index=True)

    '''    
    if outfp is not None:
        processedObject = pd.read_csv(outfp)
        processedObject = processedObject.sort_values('timestamp')
        processedObject.to_csv(outfp, index=False)
    '''

    # might need to add [:sample_count]
    return None if (outfp is not None) else outframe.sort_values('timestamp')



def _get_bars_dynamic_second_data( fp, factor, bar_type, chunksize=1000000, outfp=None, rowModifier=None):
    cols = ["timestamp", "open", "high", "low", "close", "volume", "tick_num", "num_ticks_sampled"]
    overlap = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    chunknum = 0
    outdata = np.zeros(shape=(0, 8))
    sample_count = 0
    rolling_val = 0
    open_tick = 0
    numpyOpenTick = 0
    num_rows = 0
    #last_sample_tick = 0
    loadRows = rowModifier
    if rowModifier == None:
        loadRows = sys.maxsize
    
    outframe = pd.DataFrame(columns=cols)
    elapsedBars = 0
    sample_thresh = factor

    for chunk in tqdm(pd.read_csv(fp, chunksize=chunksize, nrows=loadRows), desc='chunk #'):
        chunknum += 1
        #chunk.rename(columns={"sip_timestamp": "timestamp", "price": "price", "size": "volume"}, inplace=True)
        chunk.rename(columns={"t": "timestamp",'o': 'open', 'h':'high', 'l':'low', "c": "close", "v": "volume"}, inplace=True)
        chunk = pd.concat([overlap, chunk], axis=0)

        # if converting to datetime use np not pandas?
        #ts = pd.to_datetime(chunk['timestamp'], unit='ns').to_numpy() # might need to convert to datetime first
        ts = chunk['timestamp'].to_numpy()
        price = chunk['close'].to_numpy()
        vol = chunk['volume'].to_numpy()
        openPrice = chunk['open'].to_numpy()
        highPrice = chunk['high'].to_numpy()
        lowPrice = chunk['low'].to_numpy()
        approx_dv = np.dot(price, vol)
        # could also try using a 2 month delayed market cap
        if bar_type == 'dollar':# and factor < 100:
            #dynamicThresh = approx_dv / ((ts[-1] - ts[0]) / 8.64E13)
            #dynamicThresh = dynamicThresh * factor
            timePeriod = (pd.to_datetime(chunk['timestamp'].iloc[-1], unit='ms') - pd.to_datetime(chunk['timestamp'].iloc[0], unit='ms')).days
            rollPeriod = math.ceil((1/(timePeriod+1)) * 50000)
            volatilityEst = chunk['close'].pct_change().rolling(rollPeriod).std().fillna(method='backfill')
            
        #approx_dv = (chunk['price'] * chunk['volume']).sum()
        # still wrong, need to include sample count
        #num_rows = len(outdata) + math.ceil(approx_dv/sample_thresh)
        
        #outdata.resize((num_rows, 8), refcheck=False)
        
        outdata = {key : [] for key in cols}
        chunkSamples = 0

        offsetSplit = chunk.index.values[0]
        numpyOpenTick = open_tick - offsetSplit
        # create bars for chunk
        for i in chunk.index:
            if chunknum == 3:
                #print(i)
                xsf = 0
            #numpyIndex = np.where(chunk.index.values == i)
            pandasIndex = i
            numpyIndex = i - offsetSplit

            
            dynamicThresh = (approx_dv/timePeriod) *(0.1*(1/(1-abs(volatilityEst.loc[i]*100))))
            #sample_thresh = dynamicThresh
            

            if chunknum ==1:
                sample_thresh = factor
            else:
                sample_thresh = nextThresh

            nextThresh = dynamicThresh

            #numpyOpenTick = np.where(chunk.index.values == open_tick)

            if bar_type == 'dollar':
                rolling_val += vol[numpyIndex] * price[numpyIndex]
            elif bar_type == 'volume':
                rolling_val += vol[numpyIndex]
            elif bar_type == 'tick':
                # could be issue here, tick bars have issue
                rolling_val = numpyIndex - (sample_count * sample_thresh)

            # maybe want to use np.append instead of constantly resizing array
            # if i >= sample_vol create bar:
            if rolling_val >= sample_thresh:
                outdata[cols[0]].append(ts[numpyIndex] )                     # time
                outdata[cols[1]].append(openPrice[numpyOpenTick] )   
                #print("openTick", open_tick, "numpyOpen", numpyOpenTick, "offset",offsetSplit)
                #print(numpyIndex)
                #print(len(chunk))           # open
                outdata[cols[2]].append(np.max(highPrice[numpyOpenTick:numpyIndex+1]))     # high
                outdata[cols[3]].append(np.min(lowPrice[numpyOpenTick:numpyIndex+1]))     # low
                outdata[cols[4]].append(price[numpyIndex])                       # close
                outdata[cols[5]].append(np.sum(vol[numpyOpenTick:numpyIndex+1]))    # volume
                outdata[cols[6]].append(i)                            #tick_num
                outdata[cols[7]].append( i+1  - open_tick )              # num_ticks_sampled
                sample_count += 1
                chunkSamples += 1
                # reset volume and define next bar open tick index
                open_tick = i+1
                numpyOpenTick = open_tick - offsetSplit
                rolling_val = 0
                #last_sample_tick = i  
                #row = [ts[i], price[open_tick],  np.max(price[open_tick:i+1]), np.min(price[open_tick:i+1]),
                 #      price[i], np.sum(vol[open_tick:i+1]), i , i+1  - open_tick   ]

                #outdata.append(row)
        # no issue, iloc goes specifically by row number. use loc for exact index
        
        overlap = chunk.iloc[numpyOpenTick:] 
        #reset open tick
        #open_tick = 0    
        rolling_val = 0 

        if outfp is not None:
            if chunknum == 1:
                #pd.DataFrame.from_dict(outdata).to_csv(outfp, index=False, header=True)
                pd.DataFrame(columns=cols).to_csv(outfp, index=False, header=True)
            else:
               pd.DataFrame.from_dict(outdata).to_csv(outfp, mode='a', index=False, header=False)
            #outdata = np.empty((num_rows, 8))
            #num_rows = 0    
            #sample_count = 0
        else:
            outframe = pd.concat([outframe, pd.DataFrame.from_dict(outdata)], ignore_index=True)

    '''    
    if outfp is not None:
        processedObject = pd.read_csv(outfp)
        processedObject = processedObject.sort_values('timestamp')
        processedObject.to_csv(outfp, index=False)
    '''

    # might need to add [:sample_count]
    return None if (outfp is not None) else outframe.sort_values('timestamp')



def _get_bars_dynamic_no_lookahead_time_resample( fp, factor, bar_type, chunksize=1000000, outfp=None, rowModifier=None):
    cols = ["timestamp", "open", "high", "low", "close", "volume", "tick_num", "num_ticks_sampled"]
    overlap = pd.DataFrame(columns=['timestamp', 'price', 'volume'])
    #overlap = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    chunknum = 0
    outdata = np.zeros(shape=(0, 8))
    sample_count = 0
    rolling_val = 0
    open_tick = 0
    numpyOpenTick = 0
    num_rows = 0
    #last_sample_tick = 0
    loadRows = rowModifier
    if rowModifier == None:
        loadRows = sys.maxsize
    
    outframe = pd.DataFrame(columns=cols)
    elapsedBars = 0
    sample_thresh = factor

    for chunk in tqdm(pd.read_csv(fp, chunksize=chunksize, nrows=loadRows), desc='chunk #'):
        chunknum += 1
        chunk.rename(columns={"sip_timestamp": "timestamp", "price": "price", "size": "volume"}, inplace=True)
        

        chunk["timestamp"] = pd.to_datetime(
        chunk["timestamp"], unit="ns", utc=True
        )

        # Convert to Eastern Time (ET), accounting for both EST and EDT
        chunk["timestamp"] = chunk["timestamp"].dt.tz_convert(
            "America/New_York"
        )
        tsIndex = pd.DatetimeIndex(chunk['timestamp'])
        chunk = chunk.iloc[tsIndex.indexer_between_time('9:30','16:00')]
        #chunk = chunk.set_index('timestamp',drop=True)
        #chunk = chunk.loc[chunk.set_index('timestamp').between_time('9:30','16:00')]
        #chunk = chunk.reset_index(drop=False)
        chunk['timestamp'] = chunk['timestamp'].values.tolist()

        if chunk.empty:
            continue


        chunk = pd.concat([overlap, chunk], axis=0)

        # if converting to datetime use np not pandas?
        #ts = pd.to_datetime(chunk['timestamp'], unit='ns').to_numpy() # might need to convert to datetime first
        ts = chunk['timestamp'].to_numpy()
        price = chunk['price'].to_numpy()
        vol = chunk['volume'].to_numpy()
        approx_dv = np.dot(price, vol)
        # could also try using a 2 month delayed market cap
        if bar_type == 'dollar':# and factor < 100:
            #dynamicThresh = approx_dv / ((ts[-1] - ts[0]) / 8.64E13)
            #dynamicThresh = dynamicThresh * factor
            timePeriod = (pd.to_datetime(chunk['timestamp'].iloc[-1]) - pd.to_datetime(chunk['timestamp'].iloc[0])).days
            rollPeriod = math.ceil((1/(timePeriod+1)) * 50000)
            volatilityEst = chunk['price'].pct_change().rolling(rollPeriod).std().fillna(method='backfill')
            
        #approx_dv = (chunk['price'] * chunk['volume']).sum()
        # still wrong, need to include sample count
        #num_rows = len(outdata) + math.ceil(approx_dv/sample_thresh)
        
        #outdata.resize((num_rows, 8), refcheck=False)
        # Convert 'participant_timestamp' to datetime (assuming nanoseconds Unix timestamp)

        
        outdata = {key : [] for key in cols}
        chunkSamples = 0

        offsetSplit = chunk.index.values[0]
        numpyOpenTick = open_tick - offsetSplit
        # create bars for chunk
        for i in chunk.index:
            if chunknum == 3:
                #print(i)
                xsf = 0
            #numpyIndex = np.where(chunk.index.values == i)
            pandasIndex = i
            numpyIndex = i - offsetSplit

            
            dynamicThresh = (approx_dv/timePeriod) *(0.1*(1/(1-abs(volatilityEst.loc[i]*100))))
            #sample_thresh = dynamicThresh
            

            if chunknum ==1:
                sample_thresh = factor
            else:
                sample_thresh = nextThresh

            nextThresh = dynamicThresh

            #numpyOpenTick = np.where(chunk.index.values == open_tick)

            if bar_type == 'dollar':
                rolling_val += vol[numpyIndex] * price[numpyIndex]
            elif bar_type == 'volume':
                rolling_val += vol[numpyIndex]
            elif bar_type == 'tick':
                # could be issue here, tick bars have issue
                rolling_val = numpyIndex - (sample_count * sample_thresh)

            # maybe want to use np.append instead of constantly resizing array
            # if i >= sample_vol create bar:
            if rolling_val >= sample_thresh:
                outdata[cols[0]].append(ts[numpyIndex] )                     # time
                outdata[cols[1]].append(price[numpyOpenTick] )   
                #print("openTick", open_tick, "numpyOpen", numpyOpenTick, "offset",offsetSplit)
                #print(numpyIndex)
                #print(len(chunk))           # open
                outdata[cols[2]].append(np.max(price[numpyOpenTick:numpyIndex+1]))     # high
                outdata[cols[3]].append(np.min(price[numpyOpenTick:numpyIndex+1]))     # low
                outdata[cols[4]].append(price[numpyIndex])                       # close
                outdata[cols[5]].append(np.sum(vol[numpyOpenTick:numpyIndex+1]))    # volume
                outdata[cols[6]].append(i)                            #tick_num
                outdata[cols[7]].append( i+1  - open_tick )              # num_ticks_sampled
                sample_count += 1
                chunkSamples += 1
                # reset volume and define next bar open tick index
                open_tick = i+1
                numpyOpenTick = open_tick - offsetSplit
                rolling_val = 0
                #last_sample_tick = i  
                #row = [ts[i], price[open_tick],  np.max(price[open_tick:i+1]), np.min(price[open_tick:i+1]),
                 #      price[i], np.sum(vol[open_tick:i+1]), i , i+1  - open_tick   ]

                #outdata.append(row)
        # no issue, iloc goes specifically by row number. use loc for exact index
        
        overlap = chunk.iloc[numpyOpenTick:] 
        #reset open tick
        #open_tick = 0    
        rolling_val = 0 

        if outfp is not None:
            if chunknum == 1:
                #pd.DataFrame.from_dict(outdata).to_csv(outfp, index=False, header=True)
                #pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).to_csv(outfp, index=False, header=True)
                pd.DataFrame(columns=cols).to_csv(outfp, index=False, header=True)
            else:
                pd.DataFrame.from_dict(outdata).to_csv(outfp, mode='a', index=False, header=False)
            #outdata = np.empty((num_rows, 8))
            #num_rows = 0    
            #sample_count = 0
        else:
            outframe = pd.concat([outframe, pd.DataFrame.from_dict(outdata)], ignore_index=True)

    '''    
    if outfp is not None:
        processedObject = pd.read_csv(outfp)
        processedObject = processedObject.sort_values('timestamp')
        processedObject.to_csv(outfp, index=False)
    '''

    # might need to add [:sample_count]
    return None if (outfp is not None) else outframe.sort_values('timestamp')



def _get_bars_dynamic_no_lookahead_sample_during_market( fp, factor, bar_type, chunksize=1000000, outfp=None, rowModifier=None):
    cols = ["timestamp", "open", "high", "low", "close", "volume", "tick_num", "num_ticks_sampled"]
    overlap = pd.DataFrame(columns=['timestamp', 'price', 'volume'])
    #overlap = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    chunknum = 0
    outdata = np.zeros(shape=(0, 8))
    sample_count = 0
    rolling_val = 0
    open_tick = 0
    numpyOpenTick = 0
    num_rows = 0
    #last_sample_tick = 0
    loadRows = rowModifier
    if rowModifier == None:
        loadRows = sys.maxsize
    
    outframe = pd.DataFrame(columns=cols)
    elapsedBars = 0
    sample_thresh = factor

    for chunk in tqdm(pd.read_csv(fp, chunksize=chunksize, nrows=loadRows), desc='chunk #'):
        chunknum += 1
        chunk.rename(columns={"sip_timestamp": "timestamp", "price": "price", "size": "volume"}, inplace=True)
        

        chunk = pd.concat([overlap, chunk], axis=0)

        # if converting to datetime use np not pandas?
        #ts = pd.to_datetime(chunk['timestamp'], unit='ns').to_numpy() # might need to convert to datetime first
        ts = chunk['timestamp'].to_numpy()
        price = chunk['price'].to_numpy()
        vol = chunk['volume'].to_numpy()
        approx_dv = np.dot(price, vol)
        # could also try using a 2 month delayed market cap
        if bar_type == 'dollar':# and factor < 100:
            #dynamicThresh = approx_dv / ((ts[-1] - ts[0]) / 8.64E13)
            #dynamicThresh = dynamicThresh * factor
            timePeriod = (pd.to_datetime(chunk['timestamp'].iloc[-1]) - pd.to_datetime(chunk['timestamp'].iloc[0])).days
            rollPeriod = math.ceil((1/(timePeriod+1)) * 50000)
            volatilityEst = chunk['price'].pct_change().rolling(rollPeriod).std().fillna(method='backfill')
            
        #approx_dv = (chunk['price'] * chunk['volume']).sum()
        # still wrong, need to include sample count
        #num_rows = len(outdata) + math.ceil(approx_dv/sample_thresh)
        
        #outdata.resize((num_rows, 8), refcheck=False)
        
        outdata = {key : [] for key in cols}
        chunkSamples = 0

        offsetSplit = chunk.index.values[0]
        numpyOpenTick = open_tick - offsetSplit
        # create bars for chunk
        for i in chunk.index:
            if chunknum == 3:
                #print(i)
                xsf = 0
            #numpyIndex = np.where(chunk.index.values == i)
            pandasIndex = i
            numpyIndex = i - offsetSplit

            
            dynamicThresh = (approx_dv/timePeriod) *(0.1*(1/(1-abs(volatilityEst.loc[i]*100))))
            #sample_thresh = dynamicThresh
            

            if chunknum ==1:
                sample_thresh = factor
            else:
                sample_thresh = nextThresh

            nextThresh = dynamicThresh

            #numpyOpenTick = np.where(chunk.index.values == open_tick)

            if bar_type == 'dollar':
                rolling_val += vol[numpyIndex] * price[numpyIndex]
            elif bar_type == 'volume':
                rolling_val += vol[numpyIndex]
            elif bar_type == 'tick':
                # could be issue here, tick bars have issue
                rolling_val = numpyIndex - (sample_count * sample_thresh)

            # maybe want to use np.append instead of constantly resizing array
            # if i >= sample_vol create bar:
            currentTime = pd.to_datetime(ts[numpyIndex], unit='ns', utc=True)
            currentTime = currentTime.tz_convert("America/New_York")

            mktOpen = currentTime.replace(hour=9,minute=30, second=0,microsecond=0,nanosecond=0)
            mktClose = currentTime.replace(hour=16,minute=0, second=0,microsecond=0,nanosecond=0)
            withinMktHours = currentTime > mktOpen and currentTime < mktClose
            if rolling_val >= sample_thresh and withinMktHours:
                outdata[cols[0]].append(ts[numpyIndex] )                     # time
                outdata[cols[1]].append(price[numpyOpenTick] )   
                #print("openTick", open_tick, "numpyOpen", numpyOpenTick, "offset",offsetSplit)
                #print(numpyIndex)
                #print(len(chunk))           # open
                outdata[cols[2]].append(np.max(price[numpyOpenTick:numpyIndex+1]))     # high
                outdata[cols[3]].append(np.min(price[numpyOpenTick:numpyIndex+1]))     # low
                outdata[cols[4]].append(price[numpyIndex])                       # close
                outdata[cols[5]].append(np.sum(vol[numpyOpenTick:numpyIndex+1]))    # volume
                outdata[cols[6]].append(i)                            #tick_num
                outdata[cols[7]].append( i+1  - open_tick )              # num_ticks_sampled
                sample_count += 1
                chunkSamples += 1
                # reset volume and define next bar open tick index
                open_tick = i+1
                numpyOpenTick = open_tick - offsetSplit
                rolling_val = 0
                #last_sample_tick = i  
                #row = [ts[i], price[open_tick],  np.max(price[open_tick:i+1]), np.min(price[open_tick:i+1]),
                 #      price[i], np.sum(vol[open_tick:i+1]), i , i+1  - open_tick   ]

                #outdata.append(row)
        # no issue, iloc goes specifically by row number. use loc for exact index
        
        overlap = chunk.iloc[numpyOpenTick:] 
        #reset open tick
        #open_tick = 0    
        rolling_val = 0 

        if outfp is not None:
            if chunknum == 1:
                #pd.DataFrame.from_dict(outdata).to_csv(outfp, index=False, header=True)
                #pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).to_csv(outfp, index=False, header=True)
                pd.DataFrame(columns=cols).to_csv(outfp, index=False, header=True)
            else:
                pd.DataFrame.from_dict(outdata).to_csv(outfp, mode='a', index=False, header=False)
            #outdata = np.empty((num_rows, 8))
            #num_rows = 0    
            #sample_count = 0
        else:
            outframe = pd.concat([outframe, pd.DataFrame.from_dict(outdata)], ignore_index=True)

    '''    
    if outfp is not None:
        processedObject = pd.read_csv(outfp)
        processedObject = processedObject.sort_values('timestamp')
        processedObject.to_csv(outfp, index=False)
    '''

    # might need to add [:sample_count]
    return None if (outfp is not None) else outframe.sort_values('timestamp')

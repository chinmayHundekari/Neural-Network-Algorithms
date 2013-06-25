import pandas as pd
import numpy as np
import scipy.io as sio
import math
import copy
import sys,csv
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkstudy.EventProfiler as ep

def processData(d_data):
    df_close = d_data['close']
    ts_market = df_close['SPY']
    ls_symbols=list(d_data['close'].columns)

    BOLLINGER = 20
    print "Calculating Bollinger Band Values..."
    means = pd.rolling_mean(d_data['close'],BOLLINGER,min_periods=BOLLINGER)
    rolling_std = pd.rolling_std(d_data['close'],BOLLINGER,min_periods=BOLLINGER)
    max_bol = means + rolling_std
    min_bol = means - rolling_std
    Bollinger = (d_data['close'] - means) / (rolling_std)

    print "Converting data to inputs for learning."

    # Creating an empty dataframe
    df_events = copy.deepcopy(df_close)
    df_events = df_events * np.NAN

    # Time stamps for the event range
    ldt_timestamps = Bollinger.index
    t_beg = -3 
    t_end = 3
    t_endl = t_end+1
    f_symprice = np.zeros(t_endl-t_beg)
    f_bolval = np.zeros(t_endl-t_beg)
    inputrows = []
    outputs = []
    c = 0
    sym = 0
    for s_sym in ls_symbols:
        for x in range(BOLLINGER, len(ldt_timestamps)-t_end+t_beg): # 3...len-1
            i = x - t_beg
            input = np.zeros(t_end-t_beg-1)
            for t in range(t_beg,t_endl): # -3 -2 -1 0 1 2 3
                t1 = t-t_beg
                f_symprice[t1] = df_close[s_sym].ix[ldt_timestamps[i + t]] #   -3 -2 -1 0 1 2 3   -3  
                f_bolval[t1] = Bollinger[s_sym].ix[ldt_timestamps[i + t]]
                #print t1, f_bolval[t1]
                if ((t1 <= 1-t_beg) and (t1 > 0)):
                    input[t1] = f_bolval[t1]
            input[0] = (f_symprice[-t_beg] - f_symprice[0])*10.0/f_symprice[0]
            if (np.isnan(sum(input))):
                continue
            if (c == 0):
                inputrows = input
            else:
                inputrows = np.vstack((inputrows,input))
            outputval  = f_symprice[t_end] > (f_symprice[0] * 1.02)
            outputs.append(outputval)
#            print sym, c, input, outputval
            c = c + 1
        print sym
        sym = sym + 1
    outputs = np.array(outputs)
    return inputrows,outputs

def getData(startDate, endDate, symbols, cache=1):
    ldt_timestamps = du.getNYSEdays(startDate, endDate, dt.timedelta(hours=16))
    if (cache  == 1):
        dataobj = da.DataAccess('Yahoo')#, cachestalltime=0)
    else:
        dataobj = da.DataAccess(('Yahoo'), cachestalltime=0)
    symbols.append('SPY')
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    print "Obtaining data"
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method = 'ffill')
        d_data[s_key] = d_data[s_key].fillna(method = 'bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)
	return d_data

if __name__ == '__main__':
    values_file = sys.argv[1]
    dt_start = dt.datetime(2008, 1, 1)
    dt_end = dt.datetime(2009, 12, 31)
    ls_symbols = da.DataAccess('Yahoo').get_symbols_from_list('sp5002012')

    d_data = getData(dt_start,dt_end,ls_symbols,1)
    X,y = processData(d_data)
    print X
    print y
    sio.savemat('bollinger_inputs.mat', {'X':X, 'y':y} )
#   trainNetwork
#   feedNetwork
#   writeOrders
#   df_events = find_events(d_data, values_file)

#   print "Creating Study"
#   ep.eventprofiler(df_events, d_data, i_lookback=20, i_lookforward=20,
#                s_filename='sp5002012q2.pdf', b_market_neutral=True, b_errorbars=True,
#                s_market_sym='SPY')

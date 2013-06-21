import urllib
import datetime
import numpy as np


def getData(instrument, fromMonth, fromDay, fromYear, toMonth, toDay, toYear):
    url = "http://ichart.finance.yahoo.com/table.csv?s=%s&a=%d&b=%d&c=%d&d=%d&e=%d&f=%d&g=d&ignore=.csv" % (instrument, fromMonth, fromDay, fromYear, toMonth, toDay, toYear)

    f = urllib.urlopen(url)
    x = []
    var = 0
    data = []
    for line in f: 
        #buff = f.readline()
        var = var + 1
        if var == 1: continue
        row = line.rstrip('\n').split(',')
        #print row
        full_date = row[0].split('-')
        #print full_date
        dates = datetime.datetime(int(full_date[0]),int(full_date[1]),int(full_date[2]))
        x.append([dates,row[4]])
        data.append(float(row[4]))
        
    datas = np.array(data)
    datas = datas.reshape(datas.size,1)
    return datas
#    print datas
    #print buff


instrument = "SBIN.NS"
fromMonth = 01
fromDay = 01
fromYear = 2010
toMonth = 12
toDay = 31
toYear = 2011
dat = getData(instrument, fromMonth, fromDay, fromYear, toMonth, toDay, toYear)
print dat
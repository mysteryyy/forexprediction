# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 17:01:09 2018

@author: admin
"""

import os
os.chdir('C:\\Users\\admin\\Documents')
from nsepy import get_history
from datetime import date
from math import*

from dateutil import parser
import datetime as dt
from datetime import timedelta
import pandas as pd
import numpy as np
from datetime import timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from keras.models import load_model
import time
import fxcmpy
from keras.models import Sequential
from keras.optimizers import SGD
from keras import initializers
from keras.layers import Dense, Activation,LSTM,Dropout
def norm1(l):
  std1 = l.mean()+4*l.std()
  std2 = l.mean()-4*l.std()
  
  range = std1-std2
  print(range)
  l = (l/abs(l))*(l -std2)/range
  return l
def attr(p): 
 k = p.dropna()
 k['dir'] = k.Close-k.Open
 

 k['bindar'] =(k.Close-k.Open)/(abs(k.Close-k.Open)+0.0001)+1
 k['bindar'] =  round(k.bindar/2)*2 
 k['ret'] = ((k.Close.shift(-10)-k.Close)/k.Close)*100
 k['nret'] = norm1(k.ret)
 k['highspike'] = k.High - k.Open * k.bindar/2 + (k.bindar-2)/2 * k.Close
 
 k['lowspike'] = k.Close * k.bindar/2 - (k.bindar-2)/2 * k.Open - k.Low
 k['normdir'] = norm1(k['highspike']/(k['lowspike']+0.000001))
 k['normhs'] = norm1(k['dir']/(k['highspike']+0.000001))
 k['normls'] = norm1(k['dir']/(k['lowspike']+0.000001))


 k['avg'] = pd.rolling_mean(k.Close,window =14)
 k['avgdif'] = k.avg-k.avg.shift(1)
 k['avgdif'] = norm1(k.avgdif)
 k['mp'] = (k.High+k.Low)/2
 k['im'] = 1*(k.mp-(pd.rolling_min(k.Low,window = 20)))/((pd.rolling_max(k.High,window=20))-(pd.rolling_min(k.Low,window = 20)))
 k['ft'] = 0.5*np.log((1+k.im)/(1-k.im))
 
 k['tt'] = norm1(k['tickqty']/k['tickqty'].shift(1))
 k = k.dropna()
 k.fillna(0)
 return k

TOKEN = "f4f2e9e527c6143f298e4c50b27e809d16b30c9a"
con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')
lst = con.get_instruments()[1:19]
losses= []
for i in  lst:
    end1= dt.datetime.now().date()
    end = dt.datetime(end1.year,end1.month,end1.day)
    st1 =[]
    for j in range(10):
      start1 = end1-timedelta(days=100)
      start=dt.datetime(start1.year,start1.month,start1.day)
      df = con.get_candles(i, period='m15',
                start=start, end=end)
      df = df.iloc[::-1]
      st1.append(df)
      end1 = start1
      end = dt.datetime(end1.year,end1.month,end1.day)
    st1 = pd.concat(st1)
    st1=st1.iloc[::-1]
    st1['Open']= st1.bidopen
    st1['Close']= st1.bidclose
    st1['Low']= st1.bidlow
    st1['High']= st1.bidhigh
    st1 = attr(st1)
    
    print(st1)
    
             
    temp1 = st1
    l11 = []
    l12 = []
    for t in range(0,len(temp1)-20):
            print(t)
            st= time.time()
            l11.append(np.array(temp1[['normdir','normhs','normls','tt']][t:t+20]))
            l12.append(np.array(temp1['nret'][t+19:t+20]))
            st1 =time.time()
            print(l11[-1])
            print('total execution time ')
            print((len(temp1)-20)*(st1-st))
            print(str(t)+ 'complete')
    l11 = np.array(l11)
    l12 = np.array(l12)
    l11a = l11[0:round(0.7*len(l11))]
    l12a = l12[0:round(0.7*len(l12))]
    lta  = l11[round(0.70*len(l11)):]
    ltb  = l12[round(0.70*len(l12)):]
    sw = i
    sg = ""
    for i in sw:
     if(i=='/'):
        continue
     else:
        sg= sg+i
    temp1.to_pickle(sg+'.pkl')
    model = Sequential()
    model.add(LSTM(20, input_shape=(20, 4),return_sequences = True))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(LSTM(10))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    sgd = SGD(lr=1, momentum=0.9, decay=0.0, nesterov=False)
    
    model.compile(loss='mean_squared_error',optimizer = 'adam')
    model.fit(l11a, l12a, epochs=100, batch_size=100,verbose = 1)
    y1 = []
    y2= []
    pr1= model.predict(lta)
    for i in pr1:
        y1.append(i[0])
    for i in ltb:
        y2.append(i[0])
    df = pd.DataFrame({'real':y2,'predicted':y1})
    df['check'] = df.real*df.predicted
    print(len(df[df.check>0])/(len(df)))
    losses.append(len(df[df.check>0])/(len(df)))
    
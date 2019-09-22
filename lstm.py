from __future__ import division, absolute_import, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import datetime as dt

#-----------
# formatting dataset date
#-----------
def format_date(date):
    date_spl = date.split(sep='-')
    hour = date_spl[-1] + ':00:00'
    full_date_time = '-'.join(date_spl[0:-1]) + ' ' + hour
    return full_date_time

#-----------
# Loading dataset
#-----------
def format_dataset(df):
    df['date'] = df.apply(lambda x: format_date(x.get(key='date')),axis=1)
    df['month'] = df.apply(lambda x: pd.Timestamp(x.get(key='date')).month, axis=1)
    df['dayofmonth'] = df.apply(lambda x: pd.Timestamp(x.get(key='date')).day, axis=1)
    df['dayofweek'] = df.apply(lambda x: pd.Timestamp(x.get(key='date')).weekday(), axis=1)
    df.drop(df.columns[0], axis=1, inplace=True)
    df.drop(["wd", "ws", "temp"], axis=1,inplace=True)
    df.rename(columns={"date": "datetime"}, inplace=True)
    export_csv = df.to_csv (r'data/no2London-Dataset.csv', index = None, header=True)

df = pd.read_csv('data/no2Hourly.csv')
format_dataset(df)
print(df.shape)
print(df.columns)

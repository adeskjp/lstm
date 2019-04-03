#coding: utf-8

import pandas as pd
import numpy as np
 
# ローソク足描写
# pip install mpl_finance
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpl_finance

df = pd.read_csv('USD_JPY_Week2.csv', index_col='DateTime', names=['Tid', 'Dealable', 'Pair', 'DateTime', 'Buy', 'Sell'], skiprows=1)

# データフレームの最初の5行を表示
print(df.head())

# 今回使わない3カラムを削除
del df['Tid']
del df['Dealable']
del df['Pair']
del df['Sell']
print(df.head())

# データフレームのサイズを確認
print(df.shape)

# データの欠損を確認
df.isnull().sum()

# データフレームのインデックスをto_datetimeで変換
df.index = pd.to_datetime(df.index)

# 1分間ごとのデータへ変換
grouped_data = df.resample('1Min', how='ohlc')
 
# 最初の5行を表示
print(grouped_data.head())

candle_data = grouped_data[1:100]
print(candle_data)

#df_ = grouped_data.copy()
df_ = candle_data.copy()
df_.index = mdates.date2num(df_.index)
candle_data = df_.reset_index().values

# ローソク足表示
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 1, 1)
#mpl_finance.candlestick_ohlc(ax, candle_data, width=2, alpha=0.5, colorup='r', colordown='b')
#mpl_finance.candlestick_ohlc(ax, candle_data["open"], candle_data["high"], candle_data["low"], candle_data["close"], width=0.9, colorup="r", colordown="b")
mpl_finance.candlestick_ohlc(ax, candle_data, width=0.9, colorup="r", colordown="b")
ax.grid()
locator = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
plt.savefig('./candlestick_day.png')

# ローソク足表示
# fig = plt.figure(figsize=(18, 9))
# ax = plt.subplot(1, 1, 1)
# candle_temp = df[450:650]
# candlestick2_ohlc(ax, candle_temp["open"], candle_temp["high"], candle_temp["low"], candle_temp["close"], width=0.9, colorup="r", colordown="b")
 
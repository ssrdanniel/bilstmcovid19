# coding=utf-8
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.autoscale(enable=True, axis='both', tight=None)
import pandas as pd
import numpy as np
from matplotlib.pyplot import MultipleLocator
import matplotlib.dates as mdate
import datetime as dt
import matplotlib as mpl
import datetime as predict_truth_line
import seaborn 
# seaborn.set(context='notebook', style ='darkgrid', palette ='deep', font ='sans-serif', font_scale = 1, color_codes = True, rc = None)

df = pd.read_csv('./daily2.csv')

date = df[['date']].values
positive = df[['positive']].values.astype(np.float32)
recover = df[['recovered']].values.astype(np.float32)
confirm = positive-recover
""" 
df2 = pd.read_csv('./time-series-19-covid-combined.csv')
is_US = (df2['Country/Region'] == 'US')
data = df2[(is_US)][['Date', 'Confirmed', 'Recovered']]
confirm = np.insert(np.diff(data.Confirmed), 0, 0)
 """

def figure_1():
    fig = plt.figure()
    plt.figure(figsize=(9,4))
    positiveIncrease = df[['positiveIncrease']].values.astype(np.float32)
    plt.ylabel('Daily Increase Cases', size=15, weight='bold')
    plt.xlabel('Date', size=15, weight='bold')
    delta3 = dt.timedelta(days=1)
    date3_1 = dt.datetime(2020, 1, 22)
    date3_2 = dt.datetime(2020, 7, 9)
    dates3 = mpl.dates.drange(date3_1, date3_2, delta3)
    ax=plt.gca()
    line1, = plt.plot(dates3, positiveIncrease, color='r',label='Test',linestyle='-', marker='o', markersize='2',antialiased=True, lw=0.5)
    line2, = plt.plot(dates3[:-30], positiveIncrease[:-30],color='#3065f2', label='Train',linestyle='-', marker='s', markersize='2',antialiased=True, lw=0.5)
    dateFmt = mpl.dates.DateFormatter('%Y-%m-%d')
    plt.legend(handles=[line2, line1], loc='upper left', bbox_to_anchor=(0, 0.98), frameon=False)
    plt.axvline(predict_truth_line.datetime(2020,6,9), ls=':', c='g', lw=1)
    ax.xaxis.set_major_formatter(dateFmt)
    # fig.autofmt_xdate(bottom=0.18)
    plt.savefig('./figure1-3.jpg', dpi=1500)

    plt.show()
    """ res = positiveIncrease
    print(len(dates2), positiveIncrease.shape)
    ax2.plot(dates2[124:], positiveIncrease[124:], color='r',label='Test',linestyle='-', marker='s', markersize='2',antialiased=True, lw=0.5)
    #plt.plot(dates2[:125], positiveIncrease[:125], color='#3065f2', label='Train',linestyle='-', marker='s', markersize='2',antialiased=True, lw=0.5)
    # plt.vlines(100, 0, 20000, colors = "r", linestyles = "dashed")
    dateFmt = mpl.dates.DateFormatter('%Y-%m-%d')
    ax2.xaxis.set_major_formatter(dateFmt)
    plt.axvline(x=np.array(dates2)[124], color='silver', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Confirmed')

    daysLoc = mpl.dates.DayLocator()
    x_major_locator = MultipleLocator(30)
    ax2.xaxis.set_major_locator(x_major_locator)
    plt.legend(loc=2)
    fig.autofmt_xdate(bottom=0.18)
    fig.subplots_adjust(left=0.18)
    plt.savefig('./figure1-1-2.jpg', dpi=1500)
    plt.show() """


def figure_2():
    fig = plt.figure()
    plt.figure(figsize=(9,4))
    ax2 = plt.gca()
    # plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams['font.size'] = 12
    from rnn2 import prediction as m1
    dates2, y1, y0 = m1('./logs_rnn2_4_64_60_30/32_lstm.pth')
    line1, = plt.plot(dates2, y0, linestyle='-', color='r', label='Truth',marker='o', markersize='2',antialiased=True, lw=0.6)
    line2, = plt.plot(dates2, y1, linestyle='-', color='g', label='RNN', marker='s', markersize='2',antialiased=True, lw=0.6)
    print((y1-y0)[:10])

    from gru import prediction as m2
    _, y2, _ = m2('./logs_gru2_4_64_60_30/133_lstm.pth')
    line3, = plt.plot(dates2, y2, linestyle='-', color='mediumorchid', label='GRU', marker='*',markersize='2',antialiased=True, lw=0.6)

    from bilstm import prediction as m3
    _, y3, _ = m3('./logs_bilstm2_4_64_60_30/6_lstm.pth')
    line4, = plt.plot(dates2, y3, linestyle='-', color='cornflowerblue', label='Bi-LSTM',marker='^',markersize='2',antialiased=True, lw=0.6)
    plt.legend(handles=[line1, line2, line3, line4], loc='upper left', bbox_to_anchor=(0, 0.98), frameon=False)

    plt.ylabel('Confirmed', size=15, weight='bold')
    plt.xlabel('Date', size=15, weight='bold')

    dateFmt = mpl.dates.DateFormatter('%Y-%m-%d')
    ax2.xaxis.set_major_formatter(dateFmt)
    # plt.xticks(rotation=90)
    # fig.autofmt_xdate(bottom=0.18)
    # fig.subplots_adjust(left=0.18)
    plt.savefig('./figure2.jpg', dpi=1500)
    plt.show()


def figure_3():
    bi_lstm = np.load('./bilstm_error_2.npy')
    lens = 400
    mae, mse, meap = bi_lstm
    meap = meap/100
    plt.figure(figsize=(9,4))
    # line1, = plt.plot(mae, label='MAE', color='g',linestyle='-', marker='o', markersize='1',antialiased=True, lw=0.5)
    
    line2, = plt.plot(mse, label='MSE      Bi-LSTM', color='r',linestyle='-', marker='o', markersize='1',antialiased=True, lw=0.5)
    
    line3, = plt.plot(meap, label='MEAP', color='c',linestyle='-', marker='o', markersize='1',antialiased=True, lw=0.5)
    first_legend = plt.legend(handles=[line2, line3], loc='upper right', bbox_to_anchor=(1, 1), frameon=False)

    ax = plt.gca().add_artist(first_legend)
   
    # plt.ylim(0, 0.4)

    gru = np.load('./gru_error_2.npy')
    mae, mse, meap = gru
    meap = meap/100
    meap += 0.05
    
   
    # line4, = plt.plot(mae, label='MAE', color='m',linestyle='-', marker='o', markersize='1',antialiased=True, lw=0.5)
    
    line5, = plt.plot(mse, label='MSE         GRU', color='mediumorchid',linestyle='-', marker='o', markersize='1',antialiased=True, lw=0.5)
    
    line6, = plt.plot(meap, label='MEAP', color='cornflowerblue',linestyle='-', marker='o', markersize='1',antialiased=True, lw=0.5)
    plt.legend(handles=[line5, line6], loc='upper right', bbox_to_anchor=(0.98, 0.88), frameon=False)

    plt.xlabel('Epoch')
    plt.ylabel('MSE / MEAP')
    plt.savefig('./figure3.jpg', dpi=1000)
    plt.show()
    
    

# figure_1()
# figure_2()
figure_3()
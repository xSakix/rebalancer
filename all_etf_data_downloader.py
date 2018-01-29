from pandas_datareader import data as data_reader
import pandas
import os

from pandas_datareader._utils import RemoteDataError


def load_all_data(assets, end_date, start_date, max_size=50):
    data_source = 'yahoo'
    # data_source = 'google'
    # data_source = 'quandl'
    # assets = ['WIKI/'+asset for asset in assets]
    file_open = "etf_data_open.csv"
    file_close = "etf_data_close.csv"
    file_low = "etf_data_low.csv"
    file_high = "etf_data_high.csv"
    file_adj_close = "etf_data_adj_close.csv"
    panel_data = None
    if len(assets) > max_size:
        times = int(len(assets) / max_size)
        sub_assets = load_sub_assets_list(assets, max_size, times)
        open, close, high, low, adj_close = load_panel_data_many(data_source, end_date, panel_data, start_date,
                                                                 sub_assets)
        open.to_csv(file_open)
        close.to_csv(file_close)
        high.to_csv(file_high)
        low.to_csv(file_low)
        adj_close.to_csv(file_adj_close)
    else:
        panel_data = data_reader.DataReader(assets, data_source, start_date, end_date)
        panel_data.ix['Open'].to_csv(file_open)
        panel_data.ix['Close'].to_csv(file_close)
        panel_data.ix['High'].to_csv(file_high)
        panel_data.ix['Low'].to_csv(file_low)
        panel_data.ix['Adj Close'].to_csv(file_adj_close)


def load_panel_data_many(data_source, end_date, panel_data, start_date, sub_assets):
    open, close, high, low, adj_close = None, None, None, None, None
    while len(sub_assets) > 0:
        sub = sub_assets.pop()
        try:
            if panel_data is None:
                print('creating panel with :' + str(sub))
                panel_data = data_reader.DataReader(sub, data_source, start_date, end_date)
                open = panel_data.ix['Open']
                close = panel_data.ix['Close']
                high = panel_data.ix['High']
                low = panel_data.ix['Low']
                adj_close = panel_data.ix['Adj Close']
            else:
                print('adding panel for :' + str(sub))
                panel_data = data_reader.DataReader(sub, data_source, start_date, end_date)
                open = open.join(panel_data.ix['Open'])
                close = close.join(panel_data.ix['Close'])
                high = high.join(panel_data.ix['High'])
                low = low.join(panel_data.ix['Low'])
                adj_close = adj_close.join(panel_data.ix['Adj Close'])

        except RemoteDataError:
            sub_assets.append(sub)
    return open, close, high, low, adj_close


def load_sub_assets_list(assets, max_size, times):
    sub_assets = []
    for i in range(times):
        bottom = i * max_size
        top = (i + 1) * max_size
        if top > len(assets):
            top = -1
        sub_assets.append(assets[bottom: top])
    return sub_assets


with open('etfs.txt', 'r') as fd:
    etfs = list(fd.read().splitlines())

etfs = list(set(etfs))

print(etfs)
start_date = '1993-01-29'
end_date = '2017-12-31'

load_all_data(etfs, end_date, start_date, max_size=5)
data = pandas.read_csv('etf_data_open.csv')
print(data.keys())
print(data.index)

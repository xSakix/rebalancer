import matplotlib.pyplot as plt
import rebalancer
from sp500_data_loader import load_data
import numpy as np
import itertools
import os


def interpet_results(assets, rebalance_inv, bah_inv, data, condition, dir):
    prices = []
    for key in data.keys():
        prices.append(data[key][data.index[-1]])

    # rebalancer.writeResults('REBALANCE:', data, prices, rebalance_inv)
    # rebalancer.writeResults('B&H:', data, prices, bah_inv)
    print('rebalance: %f' % rebalance_inv.history[-1])
    print('b&h: %f' % bah_inv.history[-1])

    if condition:
        for key in data.keys():
            plt.plot(data[key], color='black')
        plt.axis('off')
        plt.savefig(dir + assets[0] + '_' + assets[1] + '.png')
        plt.clf()


with open('s&p500.txt', 'r') as fd:
    stocks = list(fd.read().splitlines())

start_date = '2010-01-01'
end_date = '2017-12-12'

stock_list = list(itertools.combinations(stocks,2))
for stock in stock_list:
    stock = list(stock)
    print('simulating: ' + str(stock))
    dir = 'stock_results_50_perc/'

    file = dir + stock[0] + '_' + stock[1] + '.png'
    file2 = dir + stock[1] + '_' + stock[0] + '.png'
    if os.path.isfile(file) or os.path.isfile(file2):
        continue
    df_open, df_close, df_high, df_low, df_adj_close = load_data(stock, start_date, end_date)
    rebalance_inv, bah_inv = rebalancer.simulate(df_adj_close, df_high, df_low, crypto=False)

    condition = (rebalance_inv.history[-1] - bah_inv.history[-1]) / bah_inv.history[-1] > 0.5
    # interpet_results(stock, rebalance_inv, bah_inv, data,condition,'stock_results/')
    # condition2 = rebalance_inv.history[-1] < bah_inv.history[-1]
    interpet_results(stock, rebalance_inv, bah_inv, df_adj_close, condition, dir)

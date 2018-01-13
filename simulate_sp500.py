import matplotlib.pyplot as plt
import rebalancer
from sp500_data_loader import load_data
import numpy as np
import itertools
import os
from multiprocessing import Process


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


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def process_stock_list(stock_list):
    start_date = '2010-01-01'
    end_date = '2017-12-12'

    for stock in stock_list:
        stock = list(stock)
        print('simulating: ' + str(stock))
        dir_reb = 'stock_results_50_perc_reb/'
        dir_bah = 'stock_results_50_perc_bah/'

        if not os.path.isdir(dir_reb):
            os.makedirs(dir_reb)

        if not os.path.isdir(dir_bah):
            os.makedirs(dir_bah)

        file = stock[0] + '_' + stock[1] + '.png'
        file2 = stock[1] + '_' + stock[0] + '.png'
        if os.path.isfile(dir_reb + file) or os.path.isfile(dir_reb + file2) or os.path.isfile(
                        dir_bah + file) or os.path.isfile(dir_bah + file2):
            continue
        df_open, df_close, df_high, df_low, df_adj_close = load_data(stock, start_date, end_date)
        i0, = np.shape(df_adj_close[stock[0]])
        i1, = np.shape(df_adj_close[stock[1]])
        if i0 == 0 or i1 == 0:
            continue
        rebalance_inv, bah_inv = rebalancer.simulate(df_adj_close, df_high, df_low, crypto=False)

        condition = (rebalance_inv.history[-1] - bah_inv.history[-1]) / bah_inv.history[-1] > 0.5
        if condition:
            interpet_results(stock, rebalance_inv, bah_inv, df_adj_close, condition, dir_reb)
        else:
            condition = (bah_inv.history[-1] - rebalance_inv.history[-1]) / rebalance_inv.history[-1] > 0.5
            if condition:
                interpet_results(stock, rebalance_inv, bah_inv, df_adj_close, condition, dir_bah)

def main():

    with open('s&p500.txt', 'r') as fd:
        stocks = list(fd.read().splitlines())


    stock_list = list(itertools.combinations(stocks, 2))
    stock_lists = chunkIt(stock_list, 4)

    processes = []
    for stock_list in stock_lists:
        print(stock_list)
        process = Process(target=process_stock_list, args=([stock_list]))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

if __name__ == '__main__':
    main()

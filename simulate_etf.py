import matplotlib.pyplot as plt
import rebalancer
import etf_data_loader
import numpy as np

def interpet_results(rebalance_inv, bah_inv, data):
    prices = []
    for key in data.keys():
        prices.append(data[key][data.index[-1]])

    rebalancer.writeResults('REBALANCE:', data, prices, rebalance_inv)
    rebalancer.writeResults('B&H:', data, prices, bah_inv)


    plt.plot(rebalance_inv.history, label='rebalance')
    plt.plot(bah_inv.history, label='buy & hold')
    plt.plot(bah_inv.invested_history, label='invested')
    legends = ['rebalance', 'buy & hold', 'invested'];
    plt.legend(legends, loc='upper left')
    plt.show()

etf = ['BND', 'SPY','XLK','VWO']
#etf = ['SPY']
start_date = '2010-01-01'
end_date = '2017-12-12'


df_open, df_close, df_high, df_low, df_adj_close = etf_data_loader.load_data(etf,start_date,end_date)
rebalance_inv, bah_inv = rebalancer.simulate(df_adj_close, df_high,df_low, crypto=False)
interpet_results(rebalance_inv, bah_inv, df_adj_close)

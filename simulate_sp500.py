import matplotlib.pyplot as plt
from pandas_datareader._utils import RemoteDataError

import rebalancer


def interpet_results(assets, rebalance_inv, bah_inv, data, condition, dir):
    prices = []
    for key in data.keys():
        prices.append(data[key][data.index[-1]])

    rebalancer.writeResults('REBALANCE:', data, prices, rebalance_inv)
    rebalancer.writeResults('B&H:', data, prices, bah_inv)

    if condition:
        for key in data.keys():
            plt.plot(data[key], color='black')
        plt.axis('off')
        plt.savefig(dir + assets[0] + '_' + assets[1] + '.png')
        plt.clf()


with open('s&p500.txt', 'r') as fd:
    stocks = list(fd.read().splitlines())

stocks = stocks[:10]

start_date = '2010-01-01'
end_date = '2017-12-12'

stock_list = []
for stock1 in stocks:
    for stock2 in stocks:
        if stock1 == stock2:
            continue
        if stock_list.__contains__([stock1, stock2]) or stock_list.__contains__([stock2, stock1]):
            continue
        stock_list.append([stock1, stock2])

for stock in stock_list:
    try:
        data, data2 = rebalancer.load_data(stock, end_date, start_date)
        rebalance_inv, bah_inv = rebalancer.simulate(data, data2, crypto=False)

        # condition = rebalance_inv.history[-1] > bah_inv.history[-1]
        # interpet_results(stock, rebalance_inv, bah_inv, data,condition,'stock_results/')
        condition2 = rebalance_inv.history[-1] < bah_inv.history[-1]
        interpet_results(stock, rebalance_inv, bah_inv, data, condition2, 'stock_bah_results/')
    except RemoteDataError or IndexError:
        print('error simulating:' + str(stock))

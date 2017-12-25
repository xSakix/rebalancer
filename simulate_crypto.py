import matplotlib.pyplot as plt

import rebalancer


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

crypto_list = ['BTC-USD', 'VTC-USD']
start_date = '2017-01-01'
end_date = '2017-12-12'

data, data2 = rebalancer.load_data(crypto_list, end_date, start_date)
rebalance_inv, bah_inv = rebalancer.simulate(data, data2, crypto=False)
interpet_results(rebalance_inv, bah_inv, data)

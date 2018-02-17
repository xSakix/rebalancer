import matplotlib.pyplot as plt
import rebalancer
import numpy as np
import sys

sys.path.insert(0, '../etf_data')
from etf_data_loader import load_all_data_from_file

sys.path.insert(0, '../buy_hold_simulation')
import bah_simulator as bah

pc_list = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5]


def interpet_results(reb_inv_list, bah_inv, data):
    prices = []
    for key in data.keys():
        prices.append(data[key][data.index[-1]])

    # rebalancer.writeResults('REBALANCE:', data, prices, reb_inv)
    # print('B&H:')
    # print('invested:' + str(bah_inv.invested_history[-1]))
    # print('value gained:' + str(bah_inv.history[-1]))
    # print('returns:' + str(bah_inv.ror_history[-1]))
    # print('mean returns:' + str(bah_inv.m)+'+/-'+str(investor.std))

    # _,(ax0,ax1) = plt.subplots(2,1)
    legends = []
    for pc, reb_inv in zip(pc_list, reb_inv_list):
        # ax0.plot(reb_inv.history, label='rebalance '+str(i/100))
        plt.plot(reb_inv.history, label='rebalance ' + str(pc))
        legends.append('rebalance ' + str(pc))

    # ax0.plot(bah_inv.history, label='b&h')
    # ax0.plot(reb_inv.invested_history, label='invested')
    plt.plot(bah_inv.history, label='b&h')
    plt.plot(reb_inv.invested_history, label='invested')
    legends.append('b&h')
    legends.append('invested')
    # ax0.legend(legends, loc='upper left')
    plt.legend(legends, loc='upper left')

    # for m in rebalance_inv.rms_list:
    #     ax1.plot(m)

    # ax1.plot(rebalance_inv.pc)
    # ax1.plot(reb_inv_list[0].apc)
    # ax1.plot(reb_inv_list[0].vol)
    # ax1.legend(['apc','vol'])

    plt.show()


start_date = '1993-01-01'
end_date = '2017-12-31'
df_adj_close = load_all_data_from_file('etf_data_adj_close.csv', start_date, end_date)

# etf = ['BND', 'SPY','XLK','VWO']
etf = ['SPY']
df_adj_close = df_adj_close[etf]

reb_inv_list = []

for pc in pc_list:
    print(pc)
    rebalance_inv = rebalancer.simulate(df_adj_close, pc=pc)
    reb_inv_list.append(rebalance_inv)

dca = bah.DCA(30, 300.)
investor = bah.Investor(etf, [1.0], dca)
sim = bah.BuyAndHoldInvestmentStrategy(investor, 2.)
sim.invest(df_adj_close)
investor.compute_means()

interpet_results(reb_inv_list, investor, df_adj_close)

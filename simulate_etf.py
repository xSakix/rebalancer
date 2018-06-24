import sys

import matplotlib.pyplot as plt

import rebalancer

sys.path.insert(0, '../etf_data')
from etf_data_loader import load_all_data_from_file2

sys.path.insert(0, '../buy_hold_simulation')
import bah_simulator as bah

pc_list = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5]

# pc_list = [.08]


def interpet_results(reb_inv_lis, bah_inv, pc_list):
    legends = []
    for pc, reb_inv in zip(pc_list, reb_inv_list):
        plt.plot(reb_inv.history, label='rebalance ' + str(pc))
        legends.append('rebalance ' + str(pc))

    plt.plot(bah_inv.history, color='black', label='b&h')
    plt.plot(bah_inv.invested_history, label='invested')
    legends.append('b&h')
    legends.append('invested')
    plt.legend(legends, loc='upper left')
    plt.show()


start_date = '1993-01-01'
end_date = '2018-06-15'
data = load_all_data_from_file2('mil_etf_data_adj_close.csv', start_date, end_date)

etf = ['IH2O.MI',
       'INDG.MI',
       'MGT.MI',
       'BUND2L.MI',
       'XS3R.MI',
       'XTXC.MI',
       'SWDA.MI',
       'DJE.MI',
       'IUSE.MI'
       ]
df_adj_close = data[etf]

plt.plot(df_adj_close)
plt.legend(df_adj_close)
plt.show()

reb_inv_list = []

for pc in pc_list:
    print(pc)
    rebalance_inv = rebalancer.simulate(df_adj_close, pc=pc)
    reb_inv_list.append(rebalance_inv)
    rebalancer.writeResults('reb', etf, df_adj_close.tail(1).as_matrix()[0], rebalance_inv)

dca = bah.DCA(30, 300.)
investor = bah.Investor(etf, [1.0], dca)
sim = bah.BuyAndHoldInvestmentStrategy(investor, 2.)
sim.invest(df_adj_close, tickets=etf)

print('B&H')
print('zisk:', investor.history[-1])
print('ror:', investor.ror_history[-1])

interpet_results(reb_inv_list, investor, pc_list)

# dca = bah.DCA(30, 300.)
# investor = bah.Investor(etf[1], [1.0], dca)
# sim = bah.BuyAndHoldInvestmentStrategy(investor, 2.)
# sim.invest(data[etf[1]], [etf[1]])
#
# print('B&H')
# print('zisk:', investor.history[-1])
# print('ror:', investor.ror_history[-1])
#
# interpet_results(reb_inv_list, investor, pc_list)

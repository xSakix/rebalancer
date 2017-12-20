from pandas_datareader import data as data_reader
import pandas
import matplotlib.pyplot as plt
import os
import numpy as np


class Investor:
    def __init__(self):
        self.cash = 0.
        self.invested = 0.
        self.history = []
        self.invested_history = []
        self.shares = []
        self.rebalances = 0


class RebalancingInvestmentStrategy:
    def __init__(self, investor, dist, tr_cost, crypto=False):
        self.investor = investor
        self.dist = dist
        self.tr_cost = tr_cost
        self.crypto = crypto

    def invest(self, data, data2):
        if len(data.keys()) == 0:
            return

        self.investor.shares = np.zeros(len(data.keys()), dtype='float64')

        for day in range(len(data[next(iter(dict(data)))])):
            if day % 30 == 0:
                self.investor.cash += 300.
                self.investor.invested += 300.

            prices = []
            close = []
            for key in data.keys():
                prices.append(data[key][day])
                close.append(data2[key][day])

            prices = np.array(prices, dtype='float64')
            portfolio = self.investor.cash + np.dot(prices, self.investor.shares)

            self.investor.history.append(portfolio)
            self.investor.invested_history.append(self.investor.invested)

            vals = np.multiply(prices, self.investor.shares)
            distribution = np.divide(vals, portfolio)
            distribution = np.abs(np.subtract(distribution, self.dist))

            rebalance = False
            for d in distribution:
                if d > 0.1:
                    rebalance = True
                    break

            std = np.abs(np.subtract(prices, close))
            noise = np.random.normal(0, std)
            prices = np.add(prices, noise)

            if day % 30 == 0 or rebalance:
                c = np.multiply(self.dist, portfolio)

                if not self.crypto:
                    c = np.subtract(c, self.tr_cost)
                    s = np.divide(c, prices)
                    s = np.floor(s)
                else:
                    cost = np.multiply(c, self.tr_cost)
                    c = np.subtract(c, cost)
                    s = np.divide(c, prices)
                self.investor.shares = s
                self.investor.rebalances += 1
                if not self.crypto:
                    self.investor.cash = portfolio - np.dot(self.investor.shares, prices) - 3 * self.tr_cost
                else:
                    value = np.dot(self.investor.shares, prices)
                    cost = np.multiply(c, self.tr_cost)
                    self.investor.cash = portfolio - value - np.sum(cost)


class BuyAndHoldInvestmentStrategy:
    def __init__(self, investor, dist, tr_cost, crypto=False):
        self.investor = investor
        self.dist = dist
        self.tr_cost = tr_cost
        self.crypto = crypto

    def invest(self, data, data2):
        if len(data.keys()) == 0:
            return

        self.investor.shares = np.zeros(len(data.keys()))

        for day in range(len(data[next(iter(dict(data)))])):
            if day % 30 == 0:
                self.investor.cash += 300.
                self.investor.invested += 300.

            prices = []
            close = []
            for key in data.keys():
                prices.append(data[key][day])
                close.append(data2[key][day])

            prices = np.array(prices)
            close = np.array(close)

            portfolio = self.investor.cash + np.dot(prices, self.investor.shares)

            self.investor.history.append(portfolio)
            self.investor.invested_history.append(self.investor.invested)

            std = np.abs(np.subtract(prices, close))
            noise = np.random.normal(0, std)
            prices = np.add(prices, noise)

            if day % 30 == 0:
                c = np.multiply(self.dist, portfolio)
                if not self.crypto:
                    c = np.subtract(c, self.tr_cost)
                    s = np.divide(c, prices)
                    s = np.floor(s)
                else:
                    cost = np.multiply(c, self.tr_cost)
                    c = np.subtract(c, cost)
                    s = np.divide(c, prices)
                self.investor.shares = s
                if not self.crypto:
                    self.investor.cash = portfolio - np.dot(self.investor.shares, prices) - 3 * self.tr_cost
                else:
                    value = np.dot(self.investor.shares, prices)
                    cost = np.multiply(c, self.tr_cost)
                    self.investor.cash = portfolio - value - np.sum(cost)


def write_potfolio_results(investor, prices, data):
    c = np.multiply(investor.shares, prices)

    index = 0
    for key in data.keys():
        print('%s(%f)=%f , when price is %f' % (key, investor.shares[index], c[index], prices[index]))
        index += 1

    print('cash : ' + str(investor.cash))


def simulate(assets, start_date, end_date, crypto=False):
    data_source = 'yahoo'
    file = "data_open.csv"
    file2 = "data_close.csv"

    has_to_load_data = False

    if os.path.isfile(file):
        data = pandas.read_csv(file)
        for asset in assets:
            if not data.keys().contains(asset):
                has_to_load_data = True

    if not os.path.isfile(file) or has_to_load_data:
        panel_data = data_reader.DataReader(assets, data_source, start_date, end_date)
        panel_data.to_frame().to_csv('all_data.csv')
        panel_data.ix['Open'].to_csv(file)
        panel_data.ix['Close'].to_csv(file2)

    data = pandas.read_csv(file)
    data2 = pandas.read_csv(file2)

    if data['Date'][0] > data['Date'][len(data['Date']) - 1]:
        rows = []
        rows2 = []
        for i in reversed(data.index):
            row = [data[key][i] for key in data.keys()]
            row2 = [data2[key][i] for key in data2.keys()]
            rows.append(row)
            rows2.append(row2)

        data = pandas.DataFrame(rows, columns=data.keys())
        data2 = pandas.DataFrame(rows2, columns=data2.keys())

    print('Simulation from %s to %s' % (start_date, end_date))

    del data['Date']

    index = 1
    for key in data.keys():
        plt.subplot(2, len(data.keys()), index)
        plt.plot(data[key])
        plt.title(key)
        index += 1

    rebalance_inv = Investor()
    bah_inv = Investor()
    bah_investors = []

    dist = np.full(len(assets), 0.9 / len(assets))
    print(dist)
    tr_cost = 2.0
    if crypto:
        tr_cost = 0.0025

    rebalance = RebalancingInvestmentStrategy(rebalance_inv, dist, tr_cost, crypto)
    bah = BuyAndHoldInvestmentStrategy(bah_inv, dist, tr_cost, crypto)

    rebalance.invest(data, data2)
    bah.invest(data, data2)

    prices = []
    for key in data.keys():
        prices.append(data[key][len(data[key]) - 1])

    for asset in assets:
        investor = Investor()
        bah_asset = BuyAndHoldInvestmentStrategy(investor, [1.0], tr_cost, crypto)
        bah_investors.append(bah_asset)
        d1 = pandas.DataFrame(data[asset], columns=[asset])
        d2 = pandas.DataFrame(data2[asset], columns=[asset])
        bah_asset.invest(d1, d2)
        writeResults(asset, d1, prices, investor)

    writeResults('REBALANCE:', data, prices, rebalance_inv)
    writeResults('B&H:', data, prices, bah_inv)

    plt.subplot2grid((2, len(data.keys())), (1, 0), colspan=3)
    plt.plot(rebalance_inv.history, label='rebalance')
    plt.plot(bah_inv.history, label='buy & hold')
    plt.plot(bah_inv.invested_history, label='invested')
    plt.legend(('rebalance', 'buy & hold', 'invested'), loc='upper left')
    plt.show()


def writeResults(type, data, prices, rebalance_inv):
    print(type)
    write_investor_results(rebalance_inv)
    write_potfolio_results(rebalance_inv, prices, data)


def write_investor_results(rebalance_inv):
    print('Podiely:')
    print('zisk: ' + str(rebalance_inv.history[-1]))
    print('investovane: ' + str(rebalance_inv.invested))
    print('pocet rebalance: ' + str(rebalance_inv.rebalances))


# etf = ['FAB', 'UUP']
# start_date = '2011-06-16'
# end_date = '2017-12-12'

etf = ['BTC-USD', 'VTC-USD']
start_date = '2017-01-01'
end_date = '2017-12-12'

simulate(etf, start_date, end_date, crypto=False)

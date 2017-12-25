from pandas_datareader import data as data_reader
import pandas
import matplotlib.pyplot as plt
import os
import numpy as np
from pandas_datareader._utils import RemoteDataError


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

        day = 0

        for i in data.index:

            if day % 30 == 0:
                self.investor.cash += 300.
                self.investor.invested += 300.

            prices = []
            close = []
            for key in data.keys():
                prices.append(data[key][i])
                close.append(data2[key][i])

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

            prices = compute_prices(close, prices)

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

            day += 1


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

        day = 0

        for i in data.index:

            if day % 30 == 0:
                self.investor.cash += 300.
                self.investor.invested += 300.
                self.investor.rebalances += 1

            prices = []
            close = []
            for key in data.keys():
                prices.append(data[key][i])
                close.append(data2[key][i])

            prices = np.array(prices)
            close = np.array(close)

            portfolio = self.investor.cash + np.dot(prices, self.investor.shares)

            self.investor.history.append(portfolio)
            self.investor.invested_history.append(self.investor.invested)

            prices = compute_prices(close, prices)

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

            day += 1


def compute_prices(close, prices):
    std = np.abs(np.subtract(prices, close))
    for i in range(len(std)):
        if std[i] <= 0. or np.isnan(std[i]):
            std[i] = 0.001
    noise = np.random.normal(0, std)
    prices = np.add(prices, noise)
    return prices


def write_potfolio_results(investor, prices, data):
    c = np.multiply(investor.shares, prices)

    index = 0
    for key in data.keys():
        print('%s(%f)=%f , when price is %f' % (key, investor.shares[index], c[index], prices[index]))
        index += 1

    print('cash : ' + str(investor.cash))


def simulate(data, data2, crypto=False):
    if len(data) == 0:
        return

    rebalance_inv = Investor()
    bah_inv = Investor()

    dist = np.full(len(data.keys()), 0.9 / len(data.keys()))
    print(dist)
    tr_cost = 2.0
    if crypto:
        tr_cost = 0.0025

    rebalance = RebalancingInvestmentStrategy(rebalance_inv, dist, tr_cost, crypto)
    bah = BuyAndHoldInvestmentStrategy(bah_inv, dist, tr_cost, crypto)

    rebalance.invest(data, data2)
    bah.invest(data, data2)

    return rebalance_inv, bah_inv


def load_data(assets, end_date, start_date):
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
    del data2['Date']
    indexes = []
    for key in data.keys():
        for i in data[key].index:
            val = data[key][i]
            try:
                if np.isnan(val) and not indexes.__contains__(i):
                    indexes.append(i)
            except TypeError:
                if not indexes.__contains__(i):
                    indexes.append(i)
    data.drop(indexes, inplace=True)
    data2.drop(indexes, inplace=True)
    return data, data2


def writeResults(type, data, prices, investor):
    print(type)
    write_investor_results(investor)
    write_potfolio_results(investor, prices, data)


def write_investor_results(rebalance_inv):
    print('Podiely:')
    print('zisk: ' + str(rebalance_inv.history[-1]))
    print('investovane: ' + str(rebalance_inv.invested))
    print('pocet rebalance: ' + str(rebalance_inv.rebalances))

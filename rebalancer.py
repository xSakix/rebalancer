from pandas_datareader import data as data_reader
import pandas
import matplotlib.pyplot as plt
import os
import numpy as np
from pandas_datareader._utils import RemoteDataError


def load_high_low(df_high, df_low):
    if df_high is not None and df_low is not None:
        df_high_low = np.abs(df_high - df_low)

    return df_high_low

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

    def invest(self, df_open,  df_high = None, df_low = None):
        df_high_low = load_high_low(df_high, df_low)


        if len(df_open.keys()) == 0:
            return

        self.investor.shares = np.zeros(len(df_open.keys()), dtype='float64')

        day = 0

        for i in df_open.index:

            if day % 30 == 0:
                self.investor.cash += 300.
                self.investor.invested += 300.

            prices = df_open.loc[i]
            high_low = df_high_low.loc[i]
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

            prices = compute_prices(high_low, prices)

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

    def invest(self, df_open, df_high,df_low):
        df_high_low = load_high_low(df_high,df_low)

        if len(df_open.keys()) == 0:
            return

        self.investor.shares = np.zeros(len(df_open.keys()))

        day = 0

        for i in df_open.index:

            if day % 30 == 0:
                self.investor.cash += 300.
                self.investor.invested += 300.
                self.investor.rebalances += 1

            prices = df_open.loc[i]
            high_low = df_high_low.loc[i]
            high_low = np.array(high_low)

            portfolio = self.investor.cash + np.dot(prices, self.investor.shares)

            self.investor.history.append(portfolio)
            self.investor.invested_history.append(self.investor.invested)

            prices = compute_prices(high_low, prices)

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


def compute_prices(high_low, prices):
    high_low = np.array(high_low)
    high_low[high_low <= 0.] = 0.001
    high_low[np.isnan(high_low)] = 0.001
    noise = np.random.normal(0, high_low)
    prices = np.add(prices, noise)

    return prices


def write_potfolio_results(investor, prices, data):
    c = np.multiply(investor.shares, prices)

    index = 0
    for key in data.keys():
        print('%s(%f)=%f , when price is %f' % (key, investor.shares[index], c[index], prices[index]))
        index += 1

    print('cash : ' + str(investor.cash))


def simulate(price_data, df_high, df_low, crypto=False):
    if len(price_data.keys()) == 0:
        return

    rebalance_inv = Investor()
    bah_inv = Investor()

    if len(price_data.keys()) == 1:
        dist = np.array([0.5])
    else:
        dist = np.full(len(price_data.keys()), 0.9 / len(price_data.keys()))
    # print(dist)
    tr_cost = 2.0
    if crypto:
        tr_cost = 0.0025

    rebalance = RebalancingInvestmentStrategy(rebalance_inv, dist, tr_cost, crypto)
    bah = BuyAndHoldInvestmentStrategy(bah_inv, dist, tr_cost, crypto)

    rebalance.invest(price_data, df_high, df_low)
    bah.invest(price_data, df_high, df_low)

    return rebalance_inv, bah_inv


def writeResults(type, data, prices, investor):
    print(type)
    write_investor_results(investor)
    write_potfolio_results(investor, prices, data)


def write_investor_results(rebalance_inv):
    print('Podiely:')
    print('zisk: ' + str(rebalance_inv.history[-1]))
    print('investovane: ' + str(rebalance_inv.invested))
    print('pocet rebalance: ' + str(rebalance_inv.rebalances))

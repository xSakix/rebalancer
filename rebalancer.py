import pandas as pd
import numpy as np


class Investor:
    def __init__(self):
        self.cash = 0.
        self.invested = 0.
        self.history = []
        self.invested_history = []
        self.shares = []
        self.rebalances = 0
        self.rank = 0.
        self.ror_history = []
        self.vol = []
        self.apc = []
        self.pc = []


class RebalancingInvestmentStrategy:
    def __init__(self, investor, dist, tr_cost):
        self.investor = investor
        self.dist = dist
        self.tr_cost = tr_cost

    def invest(self, data, pc=0.1):
        if len(data.keys()) == 0:
            return

        self.investor.shares = np.zeros(len(data.keys()), dtype='float64')

        day = 0

        for i in data.index:
            prices = data.loc[i].values

            if prices.any() == 0.:
                continue

            if day % 30 == 0:
                self.investor.cash += 300.
                self.investor.invested += 300.

            portfolio = self.investor.cash + np.dot(prices, self.investor.shares)
            if np.isnan(portfolio):
                portfolio = 0.

            self.investor.history.append(portfolio)
            self.investor.invested_history.append(self.investor.invested)
            if self.investor.invested == 0:
                ror = 0
            else:
                ror = (portfolio - self.investor.invested) / self.investor.invested
            self.investor.ror_history.append(ror)

            rebalance = False

            vals = np.multiply(prices, self.investor.shares)
            distribution = np.divide(vals, portfolio)
            distribution = np.abs(np.subtract(distribution, self.dist))

            for d in distribution:
                if d > pc:
                    rebalance = True
                    break

            if rebalance:
                c = np.multiply(self.dist, portfolio)
                c = np.subtract(c, self.tr_cost)
                s = np.divide(c, prices)
                s = np.floor(s)
                self.investor.shares = s
                self.investor.rebalances += 1
                self.investor.cash = portfolio - np.dot(self.investor.shares, prices) - len(s) * self.tr_cost

            day += 1


class RebalancingCryptoInvestmentStrategy:
    def __init__(self, investor, dist, tr_cost):
        self.investor = investor
        self.dist = dist
        self.tr_cost = tr_cost

    def invest(self, data):

        if len(data.keys()) == 0:
            return

        self.investor.shares = np.zeros(len(data.keys()), dtype='float64')

        day = 0

        for i in data.index:
            prices = data.loc[i].values

            if day % 30 == 0:
                self.investor.cash += 300.
                self.investor.invested += 300.

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

            if day % 30 == 0 or rebalance:
                c = np.multiply(self.dist, portfolio)
                cost = np.multiply(c, self.tr_cost)
                c = np.subtract(c, cost)
                s = np.divide(c, prices)
                self.investor.shares = s
                self.investor.rebalances += 1
                value = np.dot(self.investor.shares, prices)
                cost = np.multiply(c, self.tr_cost)
                self.investor.cash = portfolio - value - np.sum(cost)

            day += 1

def simulate(price_data, pc=0.1, crypto=False):
    if len(price_data.keys()) == 0:
        return

    rebalance_inv = Investor()

    if len(price_data.keys()) == 1:
        dist = np.array([0.5])
    else:
        dist = np.full(len(price_data.keys()), 0.9 / len(price_data.keys()))
    print(dist)
    tr_cost = 2.0
    if crypto:
        tr_cost = 0.0025
        rebalance = RebalancingCryptoInvestmentStrategy(rebalance_inv, dist, tr_cost)
    else:
        rebalance = RebalancingInvestmentStrategy(rebalance_inv, dist, tr_cost)

    rebalance.invest(price_data, pc)

    return rebalance_inv


def writeResults(type, data, prices, investor):
    print(type)
    write_investor_results(investor)
    write_potfolio_results(investor, prices, data)


def write_potfolio_results(investor, prices, assets):
    c = np.multiply(investor.shares, prices)
    index = 0
    for key in assets:
        print('%s(%f)=%f , when price is %f' % (key, investor.shares[index], c[index], prices[index]))
        index += 1

    print('cash : ' + str(investor.cash))
    print('returns: ' + str(investor.ror_history[-1]))


def write_investor_results(rebalance_inv):
    print('Podiely:')
    print('zisk: ' + str(rebalance_inv.history[-1]))
    print('investovane: ' + str(rebalance_inv.invested))
    print('pocet rebalance: ' + str(rebalance_inv.rebalances))

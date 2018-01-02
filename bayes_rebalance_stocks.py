import rebalancer
from sp500_data_loader import load_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom


def plot_data(data1, data2, label1='asset1', label2='asset2'):
    f, (p1, p2) = plt.subplots(2, 2)
    p1[0].plot(data1, label=label1)
    p1[0].plot(data2, label=label2)
    p1[0].legend()
    p1[1].plot(pd.DataFrame(np.array(data1)).pct_change(), label=label1)
    p1[1].plot(pd.DataFrame(np.array(data2)).pct_change(), label=label2)
    p1[1].legend()
    p2[0].plot(np.log(data1), label=label1)
    p2[0].plot(np.log(data2), label=label2)
    p2[0].legend()
    p2[1].plot(np.log(pd.DataFrame(np.array(data1)).pct_change()), label=label1)
    p2[1].plot(np.log(pd.DataFrame(np.array(data2)).pct_change()), label=label2)
    p2[1].legend()
    plt.show()


def compute_rebalance_likehood(df_adj_close, df_high, df_low, max_iter=100):
    like = None
    counts = []
    best = 0.
    for i in range(max_iter):
        rebalance_inv, bah_inv = rebalancer.simulate(df_adj_close, df_high, df_low, crypto=False)

        # print(rebalance_inv.history[-1])
        # print(bah_inv.history[-1])
        # plot_results(rebalance_inv,bah_inv)

        reb = np.array(rebalance_inv.history)
        bah = np.array(bah_inv.history)

        diffs = reb - bah

        only_reb = diffs[np.where(diffs > 0)]

        only_bah = diffs[np.where(diffs <= 0)]

        count_reb = len(only_reb)
        count_bah = len(only_bah)

        if count_reb < count_bah:
            counts.append('BAH')
            continue
        counts.append('REB')
        # print('count of rebalance observations:%f' % count_reb)
        # print('count of buy and hold observations:%f' % count_bah)
        # print('rebalance prob = %f' % (count_reb / len(diffs)))

        # grid aprox
        p_grid = np.linspace(0., 1., len(diffs))
        priors = np.repeat(1., len(diffs))
        if like is not None and like.sum() > 0.:
            priors = like
        like = binom.pmf(count_reb, len(diffs), p_grid)
        like = like * priors
        s = like.sum()
        if s == 0:
            s = 0.0000001
        like = like / s
        if max(like) == 0.:
            break
        #print('maximum likehood(%d) = %f' % (i, max(like)))
        #print('mean, std = %f,%f' % (np.mean(like), np.std(like)))

        if rebalance_inv.history[-1] > best:
            best = rebalance_inv.history[-1]

    return p_grid, like, counts, best


def compute_buy_hold_likehood(df_adj_close, df_high, df_low, max_iter=100):
    like = None
    counts = []
    best = 0.
    for i in range(max_iter):
        rebalance_inv, bah_inv = rebalancer.simulate(df_adj_close, df_high, df_low, crypto=False)

        # print(rebalance_inv.history[-1])
        # print(bah_inv.history[-1])
        # plot_results(rebalance_inv,bah_inv)

        reb = np.array(rebalance_inv.history)
        bah = np.array(bah_inv.history)

        diffs = reb - bah

        only_reb = diffs[np.where(diffs > 0)]

        only_bah = diffs[np.where(diffs <= 0)]

        count_reb = len(only_reb)
        count_bah = len(only_bah)

        if count_reb > count_bah:
            counts.append('REB')
            continue
        counts.append('BAH')
        #print('count of rebalance observations:%f' % count_reb)
        #print('count of buy and hold observations:%f' % count_bah)
        #print('b&h prob = %f' % (count_bah / len(diffs)))

        # grid aprox
        p_grid = np.linspace(0., 1., len(diffs))
        priors = np.repeat(1., len(diffs))
        if like is not None and like.sum() > 0.:
            priors = like
        like = binom.pmf(count_bah, len(diffs), p_grid)
        like = like * priors
        s = like.sum()
        if s == 0:
            s = 0.0000001
        like = like / s
        if max(like) == 0.:
            break
        #print('maximum likehood(%d) = %f' % (i, max(like)))
        #print('mean, std = %f,%f' % (np.mean(like), np.std(like)))

        if bah_inv.history[-1] > best:
            best = bah_inv.history[-1]

    return p_grid, like, counts, best


def likehood_rebalance_observations(counts, size, event, plott):
    count = counts.count(event)
    p_grid = np.linspace(0., 1., size)
    priors = np.repeat(1., size)
    like = binom.pmf(count, size, p_grid)
    like = like * priors
    like = like / like.sum()
    plott.plot(p_grid, like)
    plott.set_title('plausibility of ' + event + ' observation in sim')


start_date = '2010-01-01'
end_date = '2017-12-12'

stocks = ['AMD', 'EIX']
df_open, df_close, df_high, df_low, df_adj_close = load_data(stocks, start_date, end_date)

# plot_data(df_adj_close[stocks[0]], df_adj_close[stocks[1]], stocks[0], stocks[1])

max_iter = 100

print('computing for rebalance')
p_grid_reb, like_reb, counts_reb, best_reb = compute_rebalance_likehood(df_adj_close, df_high, df_low, max_iter)


f, (plot1, plot2) = plt.subplots(2, 2)

plot1[0].plot(p_grid_reb, like_reb)
plot1[0].set_title('plausibility of rebalance')

likehood_rebalance_observations(counts_reb, max_iter, 'REB', plot1[1])

print('computing for b&h')
p_grid_bah, like_bah, counts_bah, best_bah = compute_buy_hold_likehood(df_adj_close, df_high, df_low, max_iter)

plot2[0].plot(p_grid_bah, like_bah)
plot2[0].set_title('plausibility of b&h')

likehood_rebalance_observations(counts_bah, max_iter, 'BAH', plot2[1])

print('best rebalance:%f' % best_reb)
print('best b&h:%f' % best_bah)

plt.show()

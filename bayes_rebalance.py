import rebalancer
from sp500_data_loader import load_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom


def plot_results(rebalance_inv, bah_inv):
    f, (p1, p2) = plt.subplots(2, 2)
    p1[0].plot(rebalance_inv.history, label='rebalance')
    p1[0].plot(bah_inv.history, label='b&h')
    p1[0].plot(rebalance_inv.invested_history, label='invested')
    p1[0].legend()
    p1[1].plot(pd.DataFrame(np.array(rebalance_inv.history)).pct_change(), label='rebalance')
    p1[1].plot(pd.DataFrame(np.array(bah_inv.history)).pct_change(), label='b&h')
    p1[1].plot(pd.DataFrame(np.array(rebalance_inv.invested_history)).pct_change(), label='invested')
    p1[1].legend()
    p2[0].plot(np.log(rebalance_inv.history), label='rebalance')
    p2[0].plot(np.log(bah_inv.history), label='b&h')
    p2[0].plot(np.log(rebalance_inv.invested_history), label='invested')
    p2[0].legend()
    p2[1].plot(np.log(pd.DataFrame(np.array(rebalance_inv.history)).pct_change()), label='rebalance')
    p2[1].plot(np.log(pd.DataFrame(np.array(bah_inv.history)).pct_change()), label='b&h')
    p2[1].plot(np.log(pd.DataFrame(np.array(rebalance_inv.invested_history))).pct_change(), label='invested')
    p2[1].legend()
    plt.show()


def compute_rebalance_likehood(df_adj_close, df_high, df_low):
    like = None
    counts = []
    size = 0
    for i in range(100):
        rebalance_inv, bah_inv = rebalancer.simulate(df_adj_close, df_high, df_low, crypto=False)

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
        size += len(diffs)
        print('count of rebalance observations:%f' % count_reb)
        print('count of buy and hold observations:%f' % count_bah)
        print('rebalance prob = %f' % (count_reb / len(diffs)))

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
        print('maximum likehood(%d) = %f' % (i, max(like)))
        print('mean, std = %f,%f' % (np.mean(like), np.std(like)))
    return p_grid, like, counts, size


def compute_buy_hold_likehood(df_adj_close, df_high, df_low):
    like = None
    counts = []
    size = 0
    for i in range(100):
        rebalance_inv, bah_inv = rebalancer.simulate(df_adj_close, df_high, df_low, crypto=False)

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
        size += len(diffs)
        print('count of rebalance observations:%f' % count_reb)
        print('count of buy and hold observations:%f' % count_bah)
        print('b&h prob = %f' % (count_bah / len(diffs)))

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
        print('maximum likehood(%d) = %f' % (i, max(like)))
        print('mean, std = %f,%f' % (np.mean(like), np.std(like)))
    return p_grid, like, counts, size


def likehoor_rebalance_observations(counts, size, event):
    cReb = counts.count(event)
    p_grid = np.linspace(0., 1., size)
    # priors = np.random.uniform(0., 1., len(diffs))
    priors = np.repeat(1., size)
    like = binom.pmf(cReb, size, p_grid)
    like = like * priors
    like = like / like.sum()
    plt.clf()
    plt.title('Likehood of '+event+' observation in sim')
    plt.plot(p_grid, like)
    plt.show()


start_date = '2010-01-01'
end_date = '2017-12-12'

df_open, df_close, df_high, df_low, df_adj_close = load_data(['ABT', 'WBA'], start_date, end_date)

print('computing for rebalance')
p_grid, like, counts, size = compute_rebalance_likehood(df_adj_close, df_high, df_low)

plt.clf()
plt.title('likehood of rebalance')
plt.plot(p_grid, like)
plt.show()

likehoor_rebalance_observations(counts, 100, 'REB')

print('computing for b&h')
p_grid, like, counts, size = compute_buy_hold_likehood(df_adj_close, df_high, df_low)

plt.clf()
plt.title('likehood of b&h')
plt.plot(p_grid, like)
plt.show()

likehoor_rebalance_observations(counts, 100, 'BAH')

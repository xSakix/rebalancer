import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(etfs, start_date):
    dir = 'c:\\downloaded_data\\USD\\'
    dfs = []
    for etf in etfs:
        data = pd.read_csv(dir + str(etf) + '.csv',  names=['Date', etf])
        data = data.ix[data['Date'] > start_date]
        data.index = range(1, len(data) + 1)
        del data['Date']
        dfs.append(data)
    return pd.concat(dfs,axis=1)

data = load_data(['SPY','BND'],'2007-04-09')
sns.kdeplot(data.SPY)
plt.show()
mean = np.array([data['SPY'].mean(),data.mean()['BND']])
std = np.array([data.std()['SPY'],data.std()['BND']])
print(mean)
print(std)
n11  = np.random.normal(0, std[0],len(data))
n12  = np.random.normal(mean[0], std[0],len(data))
n21  = np.random.normal(mean[1], std[1],len(data))
n22  = np.random.normal(mean[1], std[1],len(data))
n1 = np.abs(n11-n12)
n2 = np.abs(n21-n22)
dat = pd.DataFrame({'SPY':n1, 'BND':n2}, index=range(1, len(data) + 1))

plt.plot(data.SPY)
#plt.plot(data['SPY']+n1)
plt.plot(np.convolve(data.SPY,v=30)/30.)
plt.legend(['spy','conv SPY'])
plt.show()

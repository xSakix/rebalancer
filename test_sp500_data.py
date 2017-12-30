import numpy as np
from sp500_data_loader import load_data
import matplotlib.pyplot as plt
from scipy.stats import norm

df_open, df_close, df_high, df_low, df_adj_close = load_data(['AMZN'], '2010-01-01', '2017-12-12')
desc = df_open.describe()
print(desc.loc['mean']['AMZN'])
print(desc.loc['std']['AMZN'])
changes = np.abs(df_close - df_open) / df_open * 100.

# plt.plot(norm.pdf(df_open.index,desc.loc['mean']['AMZN'],desc.loc['std']['AMZN']))
plt.plot(changes)
plt.show()

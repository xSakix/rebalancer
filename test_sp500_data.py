import numpy as np
from sp500_data_loader import load_data
import matplotlib.pyplot as plt
from scipy.stats import norm

start_date = '2010-01-01'
end_date = '2017-12-12'


df_open, df_close, df_high, df_low, df_adj_close = load_data(['MMM','ES'], start_date, end_date)
# desc = df_open.describe()
# print(desc.loc['mean']['AMZN'])
# print(desc.loc['std']['AMZN'])
# changes = np.abs(df_close - df_open) / df_open * 100.

# plt.plot(norm.pdf(df_open.index,desc.loc['mean']['AMZN'],desc.loc['std']['AMZN']))
# plt.plot(changes)
# for i in df_open.index:
#     print(i)

plt.title('open')
plt.plot(df_open)
plt.legend(df_open.keys())
plt.show()

plt.clf()

plt.title('close')
plt.plot(df_close)
plt.legend(df_open.keys())
plt.show()

plt.clf()

plt.title('adj_close')
plt.plot(df_adj_close)
plt.legend(df_open.keys())
plt.show()

plt.clf()

plt.title('high')
plt.plot(df_high)
plt.legend(df_open.keys())
plt.show()

plt.clf()

plt.title('low')
plt.plot(df_low)
plt.legend(df_open.keys())
plt.show()


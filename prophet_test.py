#inspired by https://towardsdatascience.com/time-series-analysis-in-python-an-introduction-70d5a5b1d52a

from pandas_datareader import data as data_reader
import pandas
import fbprophet
import matplotlib.pyplot as plt

#asset = ['SPY']
asset = ['BTC-USD']
start_date = '2010-01-01'
end_date = '2018-01-24'

panel_data = data_reader.DataReader(asset, 'yahoo', start_date, end_date)
panel_data.ix['Adj Close'].to_csv('test_prophet.csv')
df_adj_close = pandas.read_csv('test_prophet.csv')
df_adj_close = df_adj_close.rename(columns={'Date': 'ds', asset[0]: 'y'})
prophet = fbprophet.Prophet()
prophet.fit(df_adj_close)

forecast = prophet.make_future_dataframe(periods=365 * 2, freq='D')
# Make predictions
forecast = prophet.predict(forecast)
prophet.plot(forecast)
plt.show()

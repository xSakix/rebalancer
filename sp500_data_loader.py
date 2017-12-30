import pandas as pd


def load_data(assets, start_date, end_date):
    df_open = load_data_from_file('sp500_data_open.csv', assets, start_date, end_date)
    df_close = load_data_from_file('sp500_data_close.csv', assets, start_date, end_date)
    df_high = load_data_from_file('sp500_data_high.csv', assets, start_date, end_date)
    df_low = load_data_from_file('sp500_data_low.csv', assets, start_date, end_date)
    df_adj_close = load_data_from_file('sp500_data_adj_close.csv', assets, start_date, end_date)
    return df_open, df_close, df_high, df_low, df_adj_close


def load_data_from_file(file, assets, start_date, end_date):
    df = pd.read_csv(file)
    df = df.loc[df.Date > start_date]
    df = df.loc[df.Date < end_date]
    df = df[assets]
    return df

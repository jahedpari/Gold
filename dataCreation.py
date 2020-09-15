import pandas as pd
from yahoofinancials import YahooFinancials


def make_initial_data(start_date, end_date):
    yahoo_financials = YahooFinancials('GC=F')
    values = yahoo_financials.get_historical_price_data(start_date, end_date, "daily")

    # Extracting the closing prices from the JSON file
    values = pd.DataFrame(values['GC=F']['prices'])[['formatted_date', 'adjclose']]
    values.columns = ['date', 'Gold']

    values['date'] = pd.to_datetime(values['date'])
    values['year'] = [d.year for d in values.date]
    values['month'] = [d.strftime('%b') for d in values.date]
    values["month"] = pd.to_datetime(values.month, format='%b', errors='coerce').dt.month

    years = values['year'].unique()

    # reordering the columns
    values = values[['date', 'year', 'month', 'Gold']]
    values = values.set_index('date')

    # remove the records which gold price is  null
    values = values.dropna()
    return values


def feature_generation(values, start_date, end_date, shift_val):
    ticker_names = ['Silver', 'Crude Oil', 'S&P500', 'MSCI EM ETF', 'Nasdaq', '10 Yr US T-Note futures',
                    '2 Yr US T-Note Futures', 'Volatility Index']
    ticker_codes = ['SI=F', 'CL=F', '^GSPC', 'EEM', '^IXIC', 'ZN=F', 'ZT=F', '^VIX']

    yahoo_financials = YahooFinancials('GC=F')
    values = values.reset_index()
    counter = 0
    for i in ticker_codes:
        # Extracting Data from Yahoo Finance, returns the output in a JSON format
        yahoo_financials = YahooFinancials(i)
        raw_data = yahoo_financials.get_historical_price_data(start_date, end_date, "daily")

        # Extracting the closing prices from the JSON file
        df = pd.DataFrame(raw_data[i]['prices'])[['formatted_date', 'adjclose']]
        df.columns = ['Date-tmp', ticker_names[counter]]
        df['Date-tmp'] = pd.to_datetime(df['Date-tmp'])

        # Adding the new instrument prices to the dataframe, Merging based on date as key
        values = values.merge(df, how='left', left_on='date', right_on='Date-tmp')
        values = values.drop(labels='Date-tmp', axis=1)
        counter += 1

    # Imputing the missing values, but considering which values were originally missing.
    cols_with_missing = [col for col in values.columns if values[col].isnull().any()]

    # Make new columns indicating what will be imputed
    for col in cols_with_missing:
        values[col + '_was_missing'] = values[col].isnull().astype(int)

    values = values.fillna(method="bfill", axis=0)

    # Calculating Short Term  Historical Returns
    imp_cols = ['Gold']
    for ticker in ticker_names:
        imp_cols.append(ticker)

    change_days = [1, 3, 5, 14, 21]
    cols = ticker_names

    for i in change_days:
        x = values[imp_cols].pct_change(periods=i).add_suffix("-T-" + str(i))
        values = pd.concat(objs=(values, x), axis=1)
        x = []

    # Calculating Long Term Historical Returns
    imp_cols = ['Gold', 'Silver', 'Crude Oil', 'S&P500', 'MSCI EM ETF']
    change_days = [60, 90, 180, 250]

    for i in change_days:
        x = values[imp_cols].pct_change(periods=i).add_suffix("-T-" + str(i))
        values = pd.concat(objs=(values, x), axis=1)
        x = []

    values['quarter'] = values['date'].dt.quarter
    values['dayofweek'] = values['date'].dt.dayofweek

    change_days = [1, 3, 5, 14, 21]

    # Lag Features
    for i in change_days:
        title = "Gold-" + str(i)
        values[title] = values.Gold.shift(periods=i)

    # Simple Moving Average (SMA)
    moving_avg = pd.DataFrame(values['date'])
    feature = 'Gold'

    # Calculating Simple Moving Average
    for window in [5, 15, 30, 50, 100, 200]:
        title = str(feature) + '-' + str(window) + '-SMA'
        moving_avg[title] = cal_SMA(values, feature, window)

    # Calculating EMA
    for window in [90, 180]:
        title = str(feature) + '-' + str(window) + '-EMA'
        moving_avg[title] = cal_EMA(values, feature, window)

    values = pd.merge(left=values, right=moving_avg, how='left', on='date')

    values = values.dropna(axis=0)

    # Adding Target
    # Calculating forward returns for Target
    y = pd.DataFrame(data=values['date'])

    # Calculates forward returns for Target with respect to the next 4 day
    title = 'Gold-T+' + str(abs(shift_val))
    y[title] = values["Gold"].shift(shift_val, axis=0)

    # Adding Target Variables
    data = pd.merge(left=values, right=y, how='inner', on='date', suffixes=(False, False))

    return data


# Calculating Simple Moving Average for
def cal_SMA(df, feature, window):
    return df[feature].rolling(window=window).mean()


# Calculating Exponential Moving Average
def cal_EMA(df, feature, window):
    return df[feature].ewm(span=window, adjust=True, ignore_na=True).mean()


def add_result(summary, to_append):
    a_series = pd.Series(to_append, index=summary.columns)
    summary = summary.append(a_series, ignore_index=True)
    summary.drop_duplicates(inplace=True, ignore_index=True, keep='first')
    return summary

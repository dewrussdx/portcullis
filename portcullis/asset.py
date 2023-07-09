'''
Asset class
Describes an asset traded on financial exchanges
'''
import yfinance as yf
import yfinance.shared as yfs
import pandas as pd


class Asset:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.ticker = None

    def __str__(self):
        return f'{self.symbol}'

    # Fill up NA values when dealing with panda dataframes
    @staticmethod
    def fill_na(df) -> object:
        if df.isna().sum().sum() > 0:
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
        return df

    # Fetch ticker information
    def get_ticker(self, force_update=False) -> yf.Ticker:
        if force_update or self.ticker is None:
            self.ticker = yf.Ticker(self.symbol)
        return self.ticker

    # Fetch timeseries for this symbol
    def get_timeseries(self, **kwargs) -> pd.DataFrame:
        print(f'Downloading timeseries for {self.symbol}...')
        data = yf.download(self.symbol, **kwargs)
        return pd.DataFrame(data) if not yfs._ERRORS.keys() else None

    # Fetch timeseries and compute returns
    def get_returns(self, **kwargs) -> pd.DataFrame:
        series = self.get_timeseries(**kwargs)
        if series is not None:
            close = series['Close']
            assert close.isna().sum().sum() == 0  # Sanity check, should not happen here
            returns = close.pct_change(
                fill_method='ffill').rename(self.symbol)
            # TODO: Figure out why some notation have pct_change multiplied by 100, we omit here: data=data[1:]*100
            return pd.DataFrame(index=series.index[1:], data=returns[1:])
        else:
            return None

    # Compute swing bounds (minimum range, mean range)
    def get_swing_bounds(self, **kwargs):
        series = self.get_timeseries(**kwargs)
        if series is not None:
            high = Asset.fill_na(series['High'])
            low = Asset.fill_na(series['Low'])
            column = high / low - 1.0
            return column.min(), column.mean()
        else:
            return None, None

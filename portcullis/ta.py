import pandas as pd
import numpy as np


# Exponentially weighted moving average

class EWMA():
    def __init__(self, span: int):
        self.span = span
        self.name = f'EWMA_{span}'

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        print('Adding Indicator:', self.name)
        df[self.name] = df['Close'].ewm(span=self.span, adjust=False).mean()
        return df


# Simple Moving Average

class SMA():
    def __init__(self, span: int):
        self.span = span
        self.name = f'SMA_{span}'

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        print('Adding Indicator:', self.name)
        df[self.name] = df['Close'].rolling(window=self.span).mean()
        return df


# Cumulative Moving Average

class CMA():
    def __init__(self):
        self.name = f'CMA'

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        print('Adding Indicator:', self.name)
        df[self.name] = df['Close'].expanding().mean()
        return df


# SMA Trendfollowing

class SMATrendFollow():
    def __init__(self, fast: int, slow: int):
        self.fast = fast
        self.slow = slow
        self.name = 'SMATrendFollow'

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        print('Adding Indicator:', self.name)
        dfc = df.copy()
        dfc['FastSMA'] = dfc['Close'].rolling(window=self.fast).mean()
        dfc['SlowSMA'] = dfc['Close'].rolling(window=self.slow).mean()
        dfc['Sig'] = np.where(dfc['FastSMA'] >= dfc['SlowSMA'], 1, 0)
        dfc['PrevSig'] = dfc['Sig'].shift(1)
        df['Buy'] = np.where((dfc['PrevSig'] == 0) & (dfc['Sig'] == 1), 1, 0)
        df['Sell'] = np.where((dfc['PrevSig'] == 1) & (dfc['Sig'] == 0), 1, 0)
        assert len(df[(df['Buy'] == 1) & (df['Sell'] == 1)]) == 0
        return df

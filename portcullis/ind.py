import pandas as pd


# Exponentially weighted moving average

class EWMA():
    def __init__(self, span: int = 8):
        self.span = span
        self.name = f'EWMA_{span}'
    def __call__(self, df: pd.DataFrame) -> (pd.DataFrame, str):
        df[self.name] = df['Close'].ewm(span=self.span, adjust=False).mean()
        return df, self.name


# Simple Moving Average

class SMA():
    def __init__(self, span: int = 8):
        self.span = span
        self.name = f'SMA_{span}'
    def __call__(self, df: pd.DataFrame) -> (pd.DataFrame, str):
        df[self.name] = df['Close']
        df[self.name].rolling(window=self.span).mean()
        return df, self.name


# Cumulative Moving Average

class CMA():
    def __init__(self):
        self.name = f'CMA'
    def __call__(self, df: pd.DataFrame) -> (pd.DataFrame, str):
        df[self.name] = df['Close'].expanding().mean()
        return df, self.name

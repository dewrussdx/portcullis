import yfinance as yf
import pandas as pd


class Stock:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.ticker = None

    def get_ticker(self, force_update=False) -> yf.Ticker:
        if force_update or self.ticker is None:
            self.ticker = yf.Ticker(self.symbol)
        return self.ticker

    def get_timeseries(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(yf.download(
            self.symbol, **kwargs))


class Portfolio:
    def __init__(self, symbols: list[str] = None):
        # Convert list of symbols to Portfolio objects
        self.storage = dict()
        self.add_symbols(symbols or [])

    def add_symbols(self, symbols: list[str]) -> None:
        for symbol in symbols:
            self.add_symbol(symbol)

    def add_symbol(self, symbol: str) -> None:
        # Check for duplicates
        assert(self.storage.get(symbol) is None)
        # Insert stock into portfolio
        self.storage[symbol.upper()] = Stock(symbol)

    def remove_symbol(self, symbol: str) -> None:
        self.storage.pop(symbol.upper(), None)

    def get_stock(self, symbol: str) -> Stock:
        return self.storage.get(symbol.upper(), None)

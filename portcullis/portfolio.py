import yfinance as yf
import pandas as pd

'''
Asset class
Describes an asset traded on financial exchanges
'''


class Asset:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.ticker = None

    def __str__(self):
        return f'{self.symbol}'

    def get_ticker(self, force_update=False) -> yf.Ticker:
        if force_update or self.ticker is None:
            self.ticker = yf.Ticker(self.symbol)
        return self.ticker

    def get_timeseries(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(yf.download(
            self.symbol, **kwargs))


'''
Portfolio class
Describes a portfolio of finanical assets
'''


class Portfolio:
    class Item:
        def __init__(self, asset, weight):
            self.asset = asset
            self.weight = weight

        def __str__(self):
            return f'{self.asset},{self.weight}'

    def __init__(self, symbols: list[str] = None):
        self._values = dict()
        self.add_items(symbols or [])

    def add_items(self, symbols: list[str]) -> None:
        # Convert list of symbols to portfolio items
        for symbol in symbols:
            self.add_item(symbol)

    def add_item(self, symbol: str) -> None:
        # Check for duplicates
        assert(self._values.get(symbol) is None)
        # Insert item into portfolio
        self._values[symbol.upper()] = Portfolio.Item(Asset(symbol), 0.0)

    def remove_item(self, symbol: str) -> None:
        self._values.pop(symbol.upper(), None)

    # Dictionary interface (partial)
    def __getitem__(self, key: str) -> Item:
        return self._values[key.upper()]

    def __iter__(self):
        return iter(self._values)

    def keys(self):
        return self._values.keys()

    def items(self):
        return self._values.items()

    def values(self):
        return self._values.values()

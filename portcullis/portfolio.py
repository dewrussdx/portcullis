import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize

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
        return pd.DataFrame(yf.download(
            self.symbol, **kwargs))

    # Fetch timeseries and compute returns
    def get_returns(self, **kwargs) -> pd.DataFrame:
        series = self.get_timeseries(**kwargs)
        data = series['Close'].pct_change(
            fill_method='ffill').rename(self.symbol)
        return pd.DataFrame(index=series.index[1:], data=data[1:] * 100)

    # Compute swing bounds (minimum range, mean range)
    def get_swing_bounds(self, **kwargs):
        series = self.get_timeseries(**kwargs)
        high = Asset.fill_na(series['High'])
        low = Asset.fill_na(series['Low'])
        column = high / low - 1.0
        return column.min(), column.mean()


'''
Portfolio class
Describes a portfolio of finanical assets
'''


class Portfolio:
    class Item:
        def __init__(self, symbol, weight=0.0):
            self.asset = Asset(symbol)
            self.weight = weight

        def __str__(self):
            return f'{self.asset} x {self.weight}'

    def __init__(self, symbols: list[str] = None):
        self._values = dict()
        self.sharp_ratio = None
        self.add_symbols(symbols or [])

    # Calculate sharp ratio
    @staticmethod
    def calculate_sharp_ratio(weights, cov, mean_return, risk_free_rate=0.0) -> float:
        expected_return = weights.dot(mean_return)
        std = np.sqrt(weights.dot(cov).dot(weights))
        return (expected_return-risk_free_rate)/std

    # Add list of assets to the portfolio
    def add_symbols(self, symbols: list[str]) -> None:
        # Convert list of symbols to portfolio items
        for symbol in symbols:
            self.add_symbol(symbol)

    # Add one asset to the portfolio
    def add_symbol(self, symbol: str) -> None:
        symbol = symbol.upper()
        # Assert no duplicates
        assert self._values.get(symbol) is None
        # Insert item into portfolio
        self._values[symbol] = Portfolio.Item(symbol)

    # Remove asset
    def remove_symbol(self, symbol: str) -> None:
        self._values.pop(symbol.upper(), None)

    # Return dataframe with returns of the portfolio
    def get_returns(self, **kwargs) -> pd.DataFrame:
        returns = None
        for _, value in self._values.items():
            df = value.asset.get_returns(**kwargs)
            if returns is None:
                returns = pd.DataFrame(index=df.index, data=df)
            else:
                returns = returns.join(df)
        return returns

    # Compute optimal portfolio (maximize sharp ratio) and return the weights
    def optimize(self, bounds=None, risk_free_rate=0.0, **kwargs) -> bool:
        D = len(self._values)
        # Set default bounds (allowing short sales) if not specified
        bounds = bounds or (-0.5, None)
        print(f'Bounds: {bounds}')
        print(f'Risk Free Rate: {risk_free_rate}')
        returns = self.get_returns(**kwargs)
        mean_return = returns.mean()
        cov = returns.cov()

        # Contraint: All weights need to add up to one
        def contraint_weights_add_up_to_one(weights):
            return weights.sum() - 1

        # Objective: Minimize the negative Sharp ratio to find the optimal portfolio
        def objective_minimize_neg_sharp_ratio(weights):
            expected_return = weights.dot(mean_return)
            # Risk (or Standard deviation) == sqrt(variance)
            std = np.sqrt(weights.dot(cov).dot(weights))
            # Negate sharp ratio (to turn mimization into maximization)
            return -(expected_return-risk_free_rate)/std

        result = minimize(
            fun=objective_minimize_neg_sharp_ratio,
            x0=np.ones(D) / D,  # Initial guess for the weights
            method='SLSQP',
            constraints=[
                {
                    'type': 'eq',
                    'fun': contraint_weights_add_up_to_one,
                }
            ],
            bounds=[bounds] * D,
        )
        if result.status == 0:
            self.sharp_ratio = -result.fun
            self._apply_weights(result.x)
            # Return sharp_ratio and portfolio weights
            return True
        return False

    # Return the Sharp retio of this portfolio
    def get_sharp_ratio(self) -> float:
        return self.sharp_ratio

    # Return weights as vector in same order as item dictionary
    def get_weights(self) -> list[float]:
        weights = []
        for _, value in self._values.items():
            weights.append(value.weight)
        return weights

    # Export timeseries to CSV file
    def export_timeseries_to_csv(self, csv, **kwargs) -> None:
        frames = []
        for _, value in self._values.items():
            asset = value.asset
            series = asset.get_timeseries(**kwargs)
            series['Name'] = asset.symbol
            frames.append(series)
        pd.concat(frames).to_csv(csv)

    # Apply weights to items
    def _apply_weights(self, weights) -> None:
        assert len(weights) == len(self._values)
        index = 0
        for _, value in self._values.items():
            value.weight = weights[index]
            index += 1

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

'''
Portfolio class
Describes a portfolio of financial assets
'''
import pandas as pd
import numpy as np
from portcullis.asset import Asset
from scipy.optimize import minimize


class Portfolio:
    class Item:
        def __init__(self, symbol, weight=0.0):
            self.asset = Asset(symbol)
            self.weight = weight

        def __str__(self):
            return "{'%s':'%s'}" % (self.asset.symbol, self.weight)

    # Constructor
    def __init__(self, symbols: list[str] = None):
        self._values = dict()
        self.sharp_ratio = None
        self.add_symbols(symbols or [])

    # Return json string
    def __str__(self):
        out = {}
        out['sharp_ratio'] = str(self.sharp_ratio)
        assets = {}
        for key, value in self._values.items():
            assets[key] = str(value.weight)
        out['assets'] = assets
        return str(out)

    # Calculate sharp ratio
    @staticmethod
    def calculate_sharp_ratio(weights, cov, mean_return, risk_free_rate=0.0) -> float:
        expected_return = weights.dot(mean_return)
        std = np.sqrt(weights.dot(cov).dot(weights))
        return (expected_return-risk_free_rate)/std

    # Fetch list of S&P 500 from wikipedia
    @staticmethod
    def fetch_sp500_symbols() -> list[str]:
        df = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        return df['Symbol'].tolist()

    # Compute top N symbols based on returns
    @staticmethod
    def filter_top_return_symbols(symbols, N, **kwargs) -> list[str]:
        pool = []
        for symbol in symbols:
            series = Asset(symbol).get_returns(**kwargs)
            if series is not None:
                pool.append((symbol, series[symbol].mean()))
        pool.sort(key=lambda x: x[1], reverse=True)
        top = []
        for i in range(min(N, len(pool))):
            top.append(pool[i][0])
        return top

    # Add multiple assets to the portfolio
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

    # Remove an asset from the portfolio
    # Note: This requires any weight lists to be refreshed, e.g. using "Portfolio.get_weights()"
    def remove_symbol(self, symbol: str) -> None:
        self._values.pop(symbol.upper(), None)

    # Return dataframe with returns of the portfolio
    def get_returns(self, **kwargs) -> pd.DataFrame:
        invalids = []
        returns = None
        for key, value in self._values.items():
            df = value.asset.get_returns(**kwargs)
            if df is None:
                invalids.append(key)
            else:
                if returns is None:
                    returns = pd.DataFrame(index=df.index, data=df)
                else:
                    returns = returns.join(df)
        if len(invalids) > 0:
            print(f'Removing invalid symbols {invalids} from portfolio...')
            for invalid in invalids:
                self.remove_symbol(invalid)
        return returns

    # Compute optimal portfolio (maximize sharp ratio) and return the weights
    def optimize(self, bounds=None, risk_free_rate=0.0, **kwargs) -> bool:
        # Set default bounds (allowing short sales) if not specified
        bounds = bounds or (0, None)
        print(f'Bounds: {bounds}')
        print(f'Risk Free Rate: {risk_free_rate}')
        returns = self.get_returns(**kwargs)
        if returns is None:
            print(f'No asset in portfolio, nothing to optimize - bailing out...')
            return False
        D = len(self._values)
        assert D > 0 and len(returns.columns) == D
        mean_return = returns.mean()
        cov = returns.cov()

        # Contraint: All weights need to add up to one
        def contraint_weights_add_up_to_one(weights):
            return weights.sum() - 1

        # Objective: Minimize the negative Sharp ratio to find the optimal portfolio
        def objective_minimize_neg_sharp_ratio(weights):
            expected_return = weights.dot(mean_return)
            # Variance
            var = weights.dot(cov).dot(weights)
            assert var >= 0
            # Risk (or Standard deviation) == sqrt(variance)
            std = np.sqrt(var)
            assert std != 0
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
            options={
                'maxiter': 100,
            }
        )

        if result.status != 0:
            print(f'*** ERROR: Optimizer failed => {result}')
            return False
        self.sharp_ratio = -result.fun
        self._apply_weights(result.x)
        return True

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
            if series:
                series['Name'] = asset.symbol
                frames.append(series)
        pd.concat(frames).to_csv(csv)

    # Apply weights to items
    def _apply_weights(self, weights) -> None:
        assert len(weights) == len(self._values)
        assert abs(sum(weights) - 1.0) < 0.000001
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

import yfinance as yf
import yfinance.shared as yfs
import pandas as pd
import numpy as np
from scipy.optimize import minimize

'''
Asset class
Describes an asset traded on financial exchanges
'''

sp500_index = ['CDW', 'GM', 'GRMN', 'STE', 'STX', 'CPRT', 'CBRE', 'CMI', 'WFC', 'MA', 'BIO', 'MDT', 'DGX', 'AKAM', 'IPG', 'EBAY', 'LMT', 'ACGL', 'ACN', 'BRO', 'CSCO', 'VTRS', 'AJG', 'ORLY', 'STT', 'AZO', 'MS', 'COF', 'ROST', 'CHTR', 'VRSK', 'WM', 'OKE', 'MDLZ', 'BXP', 'ADP', 'PTC', 'HBAN', 'CCI', 'PSA', 'PPG', 'PEAK', 'TT', 'PODD', 'MPWR', 'KIM', 'HST', 'BKNG', 'FANG', 'TAP', 'MOS', 'JBHT', 'AMAT', 'RMD', 'XEL', 'GS', 'SWKS', 'AWK', 'SBUX', 'MOH', 'FAST', 'SHW', 'BBWI', 'DPZ', 'DIS', 'FTNT', 'FMC', 'HON', 'ALLE', 'GL', 'HLT', 'PFG', 'PANW', 'EOG', 'PGR', 'WELL', 'PNR', 'WYNN', 'K', 'LVS', 'JCI', 'FDS', 'DHR', 'CAH', 'NTAP', 'ALB', 'NSC', 'FLT', 'APA', 'CI', 'AVGO', 'EW', 'HSIC', 'MO', 'WAT', 'AON', 'CPB', 'HSY', 'ITW', 'PLD', 'HAL', 'MCK', 'LIN', 'KEY', 'NCLH', 'QCOM', 'IT', 'PH', 'ETN', 'RSG', 'XOM', 'GOOGL', 'NEE', 'MMC', 'NFLX', 'AAL', 'ATVI', 'TMO', 'UPS', 'ADBE', 'ARE', 'NOC', 'EL', 'LKQ', 'FIS', 'CHD', 'FI', 'BLK', 'NRG', 'PRU', 'IEX', 'V', 'ATO', 'AXON', 'CPT', 'ODFL', 'IDXX', 'BG', 'ZION', 'BK', 'URI', 'ANET', 'NEM', 'RJF', 'AXP', 'WTW', 'TRMB', 'ADI', 'FICO', 'JNJ', 'TGT', 'BKR', 'DFS', 'ROK', 'COST', 'SO', 'RTX', 'ON', 'AMCR', 'BIIB', 'TXT', 'DUK', 'CME', 'TSCO', 'IFF', 'SYY', 'IQV', 'TEL', 'WHR', 'EQIX', 'ABC', 'F', 'META', 'PHM', 'KDP', 'SBAC', 'XRAY', 'VRTX', 'CINF', 'A', 'STZ', 'LEN', 'IVZ', 'TXN', 'STLD', 'GD', 'GPN', 'TDG', 'AMD', 'GPC', 'AMP', 'WRB', 'ENPH', 'BSX', 'ZTS', 'FSLR', 'KLAC', 'INCY', 'AOS', 'DLTR', 'ABT', 'MAA', 'MTB', 'TRV', 'YUM', 'PXD', 'WAB', 'POOL', 'MTCH', 'TPR', 'GILD', 'TSLA', 'L', 'DG', 'CRM', 'SYK', 'TECH', 'IP', 'SPG', 'VLO', 'RL', 'PFE', 'SNA', 'ABBV', 'FRT', 'LH', 'HES', 'UNH', 'INTU', 'BWA', 'PPL', 'CF', 'DVA', 'DD', 'EIX',
               'ICE', 'BAX', 'LNT', 'GE', 'HD', 'MSI', 'UNP', 'DVN', 'LYV', 'FITB', 'CMCSA', 'EA', 'FCX', 'CE', 'PCG', 'O', 'WEC', 'HUM', 'MPC', 'ISRG', 'MGM', 'APTV', 'WMT', 'CSGP', 'MRO', 'EMN', 'MKTX', 'CMG', 'ROP', 'PWR', 'JPM', 'EXPE', 'UAL', 'HRL', 'RHI', 'PAYC', 'EQT', 'NTRS', 'TYL', 'BMY', 'CL', 'TFX', 'AAP', 'LOW', 'PM', 'PAYX', 'MHK', 'PEP', 'VTR', 'EFX', 'BBY', 'CSX', 'EVRG', 'NWL', 'HCA', 'ROL', 'ED', 'AEE', 'KMB', 'UHS', 'KMI', 'PG', 'LRCX', 'ULTA', 'CVS', 'CAG', 'SEE', 'AVB', 'CMA', 'TDY', 'ESS', 'BEN', 'VMC', 'WBD', 'BR', 'ILMN', 'KMX', 'EXPD', 'HPQ', 'AME', 'HOLX', 'DHI', 'CAT', 'BAC', 'RVTY', 'ES', 'SRE', 'MRK', 'APH', 'ADSK', 'NVDA', 'MU', 'C', 'CLX', 'NOW', 'TROW', 'CB', 'DTE', 'GNRC', 'IBM', 'WST', 'APD', 'GEN', 'GOOG', 'NUE', 'LUV', 'PKG', 'MLM', 'RCL', 'TJX', 'CCL', 'DOV', 'GLW', 'SNPS', 'WY', 'KR', 'XYL', 'MNST', 'PARA', 'INTC', 'EXC', 'VFC', 'PNW', 'TTWO', 'NDSN', 'OMC', 'AVY', 'SPGI', 'NKE', 'GWW', 'CTSH', 'OXY', 'TMUS', 'CRL', 'DLR', 'LLY', 'JNPR', 'ELV', 'RF', 'MET', 'ECL', 'CNC', 'EMR', 'DRI', 'CMS', 'PSX', 'T', 'UDR', 'COO', 'CBOE', 'ZBH', 'CHRW', 'SWK', 'PNC', 'FFIV', 'MCD', 'PCAR', 'MMM', 'WBA', 'TSN', 'AMGN', 'NDAQ', 'BALL', 'SLB', 'HIG', 'BDX', 'ORCL', 'REGN', 'DXC', 'AMT', 'AFL', 'EXR', 'MAS', 'USB', 'ETR', 'EQR', 'DAL', 'HAS', 'AEP', 'NVR', 'WDC', 'FDX', 'GIS', 'TFC', 'ANSS', 'BA', 'REG', 'SJM', 'NWS', 'NI', 'LYB', 'IRM', 'DXCM', 'PEG', 'VRSN', 'CNP', 'CTAS', 'TRGP', 'NXPI', 'SCHW', 'LNC', 'EPAM', 'CTRA', 'DE', 'MTD', 'MCHP', 'WMB', 'D', 'AIZ', 'ALK', 'VZ', 'ALGN', 'RE', 'MSFT', 'AES', 'TER', 'AMZN', 'MSCI', 'HII', 'AAPL', 'CDNS', 'COP', 'ALL', 'AIG', 'NWSA', 'ZBRA', 'FE', 'ADM', 'JKHY', 'MCO', 'LHX', 'MKC', 'KO', 'LDOS', 'J', 'MAR', 'CVX']


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
            assert close.isna().sum().sum() == 0 # Sanity check, should not happen here
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


'''
Portfolio class
Describes a portfolio of financial assets
'''


class Portfolio:
    class Item:
        def __init__(self, symbol, weight=0.0):
            self.asset = Asset(symbol)
            self.weight = weight

        def __str__(self):
            return f'{self.asset} x {self.weight}'

    # Constructor
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
        bounds = bounds or (-0.5, None)
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
                'maxiter': 200,
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

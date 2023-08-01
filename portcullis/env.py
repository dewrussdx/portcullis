import numpy as np
import pandas as pd

# A very simple environment representing a stock market

State = list[float]
Action = int


class Env():
    NONE: Action = 0
    BUY: Action = 1
    SELL: Action = 2

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.actions = [Env.NONE, Env.BUY, Env.SELL]
        self.states = self.df.to_numpy(dtype=np.float32)  # Note: Optimize?
        self.maxsteps = len(self.states) - 1
        self.reset()

    def step(self, _action: Action) -> (State, float, bool):
        assert False and "Step not implemented in base environment."

    def reset(self) -> State:
        self.index = 0
        return self.states[self.index]

    def random_action(self) -> Action:
        return np.random.choice(self.actions)

    def num_actions(self) -> int:
        return len(self.actions)

    def num_features(self) -> int:
        return self.df.shape[1]

    @staticmethod
    def yf_download(symbol: str, start=None, end=None, interval='1d', features=None,
                    ta: list[any] = None, logret_col='Close'):
        import yfinance as yf
        # Download candlesticks and drop rows with invalid indices
        df = yf.download(symbol, start=start, end=end, interval=interval)
        # Keep only feature columns if specified
        if features is not None:
            df = df.loc[:, features]
        # Add indicators as specified
        ta = ta or []
        for indicator in ta:
            df = indicator(df)
        # Add log returns froms requested column
        if logret_col:
            assert logret_col in df and 'Requested column not found in dataset. Unable to compute log return.'
            df['LogReturn'] = np.log(df[logret_col]).diff()
        # Drop all rows containing any N/A columns
        df.dropna(axis=0, how='any', inplace=True)
        # Return parsed dataframe and new feature set
        return df

    @staticmethod
    def split_data(df: pd.DataFrame, test_ratio=0.25) -> (pd.DataFrame, pd.DataFrame):
        N = int(len(df)*test_ratio)
        return df.iloc[:-N].copy(), df.iloc[-N:].copy()


class DaleTrader(Env):
    def __init__(self, df: pd.DataFrame, balance: int = 50_000):
        super().__init__(df)
        self.balance = balance
        self.rewards = self.df['LogReturn'].to_numpy(dtype=np.float32)
        self.reset()

    def reset(self):
        self.invested = 0
        return super().reset()

    def step(self, action: Action) -> (State, float, bool):
        assert Env.NONE <= action <= Env.SELL
        self.index += 1
        assert self.index <= self.maxsteps

        # Calculate reward for the step
        reward = 0
        if action == Env.BUY:
            self.invested = 1
        elif action == Env.SELL:
            self.invested = 0

        # budget = min(self.balance, 1000)
        # num_shares = int(budget / self.states['Close'])
        # expense = num_shares * self.states['Close']

        reward += self.rewards[self.index] * self.invested

        # Return state
        next_state = self.states[self.index]
        done = self.index >= self.maxsteps
        return next_state, reward, done

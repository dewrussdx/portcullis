import numpy as np
import pandas as pd

# A very simple environment representing a stock market

State = list[float]
Action = int
Features = list[str]


class Env():
    NONE: Action = 0
    BUY: Action = 1
    SELL: Action = 2

    def __init__(self, df: pd.DataFrame, features: Features = None):
        self.df = df
        self.features = features or ['Close']
        self.actions = [Env.NONE, Env.BUY, Env.SELL]
        self.states = self.df[self.features].to_numpy(dtype=np.float32)
        self.rewards = self.df['Return'].to_numpy(dtype=np.float32)
        self.maxsteps = len(self.states) - 1
        self.reset()

    def num_actions(self) -> int:
        return len(self.actions)

    def reset(self) -> State:
        self.index = 0
        self.invested = False
        return self.states[self.index]

    def random_action(self) -> Action:
        return np.random.choice(self.actions)

    def step(self, action: Action) -> (State, float, bool):
        assert Env.NONE <= action <= Env.SELL
        self.index += 1
        assert self.index <= self.maxsteps
        if action == Env.BUY:
            self.invested = True
        elif action == Env.SELL:
            self.invested = False
        next_state = self.states[self.index]
        reward = self.rewards[self.index] if self.invested else 0.0
        done = self.index >= self.maxsteps
        return next_state, reward, int(done)

    @staticmethod
    def yf_download(symbol: str, start=None, end=None, interval='1d', features: Features = None, add_log_returns=True):
        import yfinance as yf
        features = features or ['Close']
        df = yf.download(symbol, start=start, end=end, interval=interval)
        df.dropna(axis=0, how='all', inplace=True)
        df.dropna(axis=1, how='any', inplace=True)
        df = df.loc[:, features]
        if add_log_returns:
            df['Return'] = np.log(df['Close']).diff()
        return df.shift(-1)

    @staticmethod
    def split_data(df: pd.DataFrame, test_ratio=0.25) -> (pd.DataFrame, pd.DataFrame):
        N = int(len(df)*test_ratio)
        return df.iloc[:-N].copy(), df.iloc[-N:].copy()

import numpy as np
import pandas as pd

State = list[float]
Action = int


class Env():

    DISCRETE = 0
    CONTINUOUS = 1

    def __init__(self, name: str, env_type: int, action_space: any, observation_space: any):
        self.name = name
        self.env_type = env_type
        self.action_space = action_space
        self.observation_space = observation_space
        self.max_steps = len(self.observation_space) - 1
        self.seed = None
        self.reset()

    def step(self, _action: Action) -> tuple[State, float, bool, bool, any]:
        pass

    def reset(self, seed: int = None) -> tuple[State, any]:
        if seed is not None:
            self.seed = seed
        self.index = 0
        return self.observation_space[self.index], None

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass

    def random_action(self) -> Action:
        return np.random.choice(self.action_space)

    def num_actions(self) -> int:
        return len(self.action_space)

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

    @staticmethod
    def get_env_type(action_space: any) -> tuple[bool, int]:
        name = type(action_space).__name__
        return (True, Env.DISCRETE) if name == 'Discrete' else (True, Env.CONTINUOUS) if name == 'Box' else (False, Env.DISCRETE)

    @staticmethod
    def get_env_spec(env: any) -> tuple[bool, int, int, int, float]:
        is_gym, env_type = Env.get_env_type(env.action_space)
        if not is_gym:
            # TODO: Native environments do not support CONTINOUS action spaces yet
            return is_gym, env_type, env.num_actions(), env.num_features(), None
        if env_type == Env.DISCRETE:
            return is_gym, env_type, env.action_space.n, env.observation_space.shape[0], None
        elif env_type == Env.CONTINUOUS:
            return is_gym, env_type, env.action_space.shape[0], env.observation_space.shape[0], env.action_space.high[0]
        assert (False)

import numpy as np
import pandas as pd

State = list[float] | float  # Alias for state
Action = int  # Alias for action


class Env():

    DISCRETE = 0  # Discrete action environment
    CONTINUOUS = 1  # Continuous action environment

    def __init__(self, name: str, env_type: int, action_space: any, observation_space: any):
        """Initialize environment.
        """
        self.name = name
        self.env_type = env_type
        self.action_space = action_space
        self.observation_space = observation_space
        self.max_steps = len(self.observation_space) - 1
        self.seed = None
        # To stay compatible with gym environments
        self._max_episode_steps = self.max_steps

    def step(self, _action: Action) -> tuple[State, float, bool, bool, any]:
        """Take a step in this environment.
        """
        pass

    def reset(self, seed: int = None) -> tuple[State, any]:
        """Reset environment.
        """
        if seed is not None:
            self.seed = seed
        self.index = 0
        return self.observation_space[self.index], None

    def render(self) -> None:
        """Render environment.
        """
        pass

    def close(self) -> None:
        """Close environment.
        """
        pass

    def random_action(self) -> Action:
        """Select a random action from the action space.
        """
        return np.random.choice(self.action_space)

    def get_action_dim(self) -> int:
        """Return the action dimensions of this environment.
        """
        return self.action_space.shape[0]

    def get_state_dim(self) -> int:
        """Return state dimension of this environment.
        """
        return self.observation_space.shape[1]

    @staticmethod
    def yf_download(symbol: str, start=None, end=None, interval='1d', features=None,
                    ta: list[any] = None, logret_col='Close'):
        """Download yfinance ticker data as pandas dataframe.
        """
        import yfinance as yf
        # Download candlesticks and drop rows with invalid indices
        df = yf.download(symbol, start=start, end=end, interval=interval)
        # Add indicators as specified
        ta = ta or []
        for indicator in ta:
            df = indicator(df)
        # Keep only feature columns if specified
        if features is not None:
            df_ret = df.loc[:, features]
        else:
            df_ret = df
        # Add log returns froms requested column
        if logret_col:
            assert logret_col in df and 'Requested column not found in dataset. Unable to compute log return.'
            df_ret['LogReturn'] = np.log(df[logret_col]).diff()
        # Drop all rows containing any N/A columns
        df_ret.dropna(axis=0, how='any', inplace=True)
        # Return parsed dataframe and new feature set
        return df_ret

    @staticmethod
    def split_data(df: pd.DataFrame, test_ratio=0.25) -> (pd.DataFrame, pd.DataFrame):
        """Split the data into training and testing data with specified ratio.
        """
        N = int(len(df)*test_ratio)
        return df.iloc[:-N].copy(), df.iloc[-N:].copy()

    @staticmethod
    def get_env_type(action_space: any) -> tuple[bool, int]:
        """Return the environment type tuple (is_gym, env_type).
        """
        name = type(action_space).__name__
        return (True, Env.DISCRETE) if name == 'Discrete' else (True, Env.CONTINUOUS) if name == 'Box' else (False, Env.DISCRETE)

    @staticmethod
    def get_env_spec(env: any) -> tuple[bool, int, int, int, float]:
        """ Return environment specs. Works with gym and portcullis native environments.
        Returns (is_gym, env_type, action_dim, state_dim, action_hi|None)
        """
        is_gym, env_type = Env.get_env_type(env.action_space)
        if not is_gym:
            # TODO: Native environments do not support CONTINOUS action spaces yet
            return False, env_type, env.get_action_dim(), env.get_state_dim(), None
        if env_type == Env.DISCRETE:
            return is_gym, env_type, env.action_space.n, env.observation_space.shape[0], None
        elif env_type == Env.CONTINUOUS:
            return is_gym, env_type, env.action_space.shape[0], env.observation_space.shape[0], env.action_space.high[0]
        assert False and "Unknown environment."

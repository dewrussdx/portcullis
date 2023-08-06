from portcullis.env import Env, State, Action
import pandas as pd
import numpy as np
import math


class DayTrader(Env):
    # Action Types
    NONE: Action = 0
    BUY: Action = 1
    SELL: Action = 2

    class Position():
        LONG = 0
        SHORT = 1
        """Class handling trading functionalty.
        """

        def __init__(self, side: int, enter_price: float):
            assert side == DayTrader.Position.LONG or side == DayTrader.Position.SHORT
            self.side = side
            self.enter_price = enter_price
            self.exit_price = 0.
            self.pnl = 0.
            self.balance = 0.
            self.num_shares = 0

        def enter(self, balance: float) -> tuple[bool, float]:
            self.balance = balance
            self.num_shares = int(math.floor(
                self.balance / self.enter_price))
            return self.num_shares > 0

        def exit(self, exit_price: float) -> float:
            self.exit_price = exit_price
            if self.is_long():
                self.pnl = (self.exit_price - self.enter_price) * \
                    self.num_shares
            else:
                self.pnl = (self.enter_price - self.exit_price) * \
                    self.num_shares
            return self.pnl

        def is_closed(self) -> bool:
            return self.exit_price > 0.

        def is_long(self) -> bool:
            return self.side == DayTrader.Position.LONG

        def is_short(self) -> bool:
            return self.side == DayTrader.Position.SHORT

        def __str__(self) -> str:
            T = 'LONG' if self.is_long() else 'SHORT'
            return f'{T}[{self.num_shares}]: PNL ${self.pnl:.2f}'

    def __init__(self, df: pd.DataFrame, balance: int = 10_000, samples_per_day=1, verbose=True):
        """DayTrading evironment.
        """
        super().__init__(name='DayTrader', env_type=Env.DISCRETE, action_space=np.array(
            [DayTrader.NONE, DayTrader.BUY, DayTrader.SELL]), observation_space=df.to_numpy(dtype=np.float32))
        self.df = df
        self.closing_prices = self.df['Close'].to_numpy(dtype=np.float32)
        self.balance = balance
        self.samples_per_day = samples_per_day
        self.verbose = verbose
        self.reset()

    def reset(self, seed: int = None) -> tuple[State, any]:
        self.current_balance = self.balance
        self.trades = []
        self.active_trade = None
        self.episode_reward = 0.
        return super().reset(seed)

    def step(self, action: Action) -> tuple[State, float, bool, bool, any]:
        """Perform a action in this environment and compute the reward, for the MDP. This function 
        returns the next state, the reward, if simulation is done or truncated. Note that to stay compatible
        with the gym environments we also return 'info' as last parameter and with None value in the tuple,
        like so: next_state, reward, done, truncated, info = env.step(action)
        """
        self.index += 1
        done, enter_trade, exit_trade = False, False, False
        reward = 0.
        close = self.closing_prices[self.index]
        if action == DayTrader.BUY:
            if self.active_trade:
                exit_trade = self.active_trade.is_short()
            else:
                self.active_trade = DayTrader.Position(
                    DayTrader.Position.LONG, enter_price=close)
                enter_trade = True
        elif action == DayTrader.SELL:
            if self.active_trade:
                exit_trade = self.active_trade.is_long()
            else:
                self.active_trade = DayTrader.Position(
                    DayTrader.Position.SHORT, enter_price=close)
                enter_trade = True
        if enter_trade:
            assert self.active_trade and not self.active_trade.is_closed()
            self.trades.append(self.active_trade)
            if not self.active_trade.enter(self.current_balance):
                done = True  # Bankrupt
        elif exit_trade:
            pnl = self.active_trade.exit(close)
            self.current_balance += pnl
            if self.current_balance <= 0.:
                done = True  # Bankrupt
            else:
                reward = pnl / self.balance
            if self.verbose:
                print(
                    f'#{self.index + 1}: {self.active_trade} => ${self.current_balance:.2f}'
                )
            self.active_trade = None

        # Return (state, reward, done)
        self.episode_reward += reward
        done = done or self.index >= self.max_steps
        if done:
            print(
                f'Done after day #{self.index}: Trades taken:{len(self.trades)}, Balance: ${self.current_balance:.2f}')
        next_state = self.observation_space[self.index]
        return next_state, reward, done, False, None


class FollowSMA(Env):
    NONE: Action = 0
    BUY: Action = 1
    SELL: Action = 2

    def __init__(self, df, features):
        super().__init__(name='FollowSMA', env_type=Env.DISCRETE, action_space=np.array([
            FollowSMA.NONE, FollowSMA.BUY, FollowSMA.SELL]), observation_space=df[features].to_numpy(dtype=np.float32))

        self.df = df
        self.features = features
        self.rewards = self.df['LogReturn'].to_numpy(dtype=np.float32)
        self.reset()

    def reset(self, seed: int = None) -> tuple[State, any]:
        self.invested = 0
        return super().reset(seed)

    def step(self, action: Action) -> tuple[State, float, bool, bool, any]:
        """Perform a action in this environment and compute the reward, for the MDP. This function 
        returns the next state, the reward, if simulation is done or truncated. Note that to stay compatible
        with the gym environments we also return 'info' as last parameter and with None value in the tuple,
        like so: next_state, reward, done, truncated, info = env.step(action)
        """
        self.index += 1
        if action == FollowSMA.BUY:
            self.invested = 1.
        elif action == FollowSMA.SELL:
            self.invested = 0.
        next_state = self.observation_space[self.index]
        reward = self.rewards[self.index] * self.invested
        done = self.index >= self.max_steps
        return next_state, reward, done, False, None


class PredictSPY(Env):
    NONE: Action = 0
    BUY: Action = 1
    SELL: Action = 2

    def __init__(self, df, features):
        super().__init__(name='PredictSPY', env_type=Env.DISCRETE, action_space=np.array([
            PredictSPY.NONE, PredictSPY.BUY, PredictSPY.SELL]), observation_space=df[features].to_numpy(dtype=np.float32))
        self.df = df
        self.features = features
        self.rewards = self.df['SPY'].to_numpy(dtype=np.float32)
        self.reset()

    def reset(self, seed: int = None) -> tuple[State, any]:
        self.invested = 0
        return super().reset(seed)

    def step(self, action: Action) -> tuple[State, float, bool, bool, any]:
        """Perform a action in this environment and compute the reward, for the MDP. This function 
        returns the next state, the reward, if simulation is done or truncated. Note that to stay compatible
        with the gym environments we also return 'info' as last parameter and with None value in the tuple,
        like so: next_state, reward, done, truncated, info = env.step(action)
        """
        self.index += 1
        if action == PredictSPY.BUY:
            self.invested = 1.
        elif action == PredictSPY.SELL:
            self.invested = 0.
        next_state = self.observation_space[self.index]
        done = self.index >= self.max_steps
        reward = self.rewards[self.index] * self.invested if not done else 0.
        return next_state, reward, done, False, None

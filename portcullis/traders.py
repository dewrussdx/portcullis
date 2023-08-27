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

        def __init__(self, side: int):
            """Initialize Trade.
            """
            assert side == DayTrader.Position.LONG or side == DayTrader.Position.SHORT
            self.side = side
            self.position_size = 0.
            self.investment = 0.
            self.enter_price = 0.
            self.exit_price = 0.
            self.pnl = 0.
            self.num_shares = 0
            self.expiration = 0

        def get_shares(self, enter_price: float, position_size: float) -> int:
            return int(math.floor(position_size / enter_price))

        def can_enter(self, balance: float, enter_price: float, position_size: float) -> bool:
            """Return whehter this trade can be entered for the given position size.
            """
            if balance >= position_size:
                return self.get_shares(enter_price, position_size) > 0
            return False

        def can_exit(self, exit_price: float, min_profit: float = None) -> bool:
            """Return whether this trade can be exited with the minimum profit specified.
            """
            if min_profit is None:
                return True
            return (self.get_pnl(exit_price) / self.position_size) >= min_profit

        def is_expired(self) -> bool:
            """Countdown hold expiration and return whether this trade is expired.
            """
            if self.expiration is None:
                return False
            self.expiration -= 1
            return self.expiration <= 0

        def enter(self, enter_price: float, position_size: float, expiration: int = None) -> float:
            """Enter this trade with given price, position size and expiration.
            """
            self.enter_price = enter_price
            self.position_size = position_size
            self.expiration = expiration
            self.num_shares = self.get_shares(
                self.enter_price, self.position_size)
            assert self.num_shares > 0
            self.investment = self.num_shares * self.enter_price
            return self.investment

        def get_pnl(self, exit_price: float) -> float:
            """Return absolute P&L after trade has been exited.
            """
            if self.is_long():
                pnl = (exit_price - self.enter_price) * self.num_shares
            else:
                pnl = (self.enter_price - exit_price) * self.num_shares
            return pnl

        def exit(self, exit_price: float) -> float:
            """Exit the trade at given price.
            """
            self.exit_price = exit_price
            self.pnl = self.get_pnl(self.exit_price)
            return self.pnl, self.investment + self.pnl

        def is_closed(self) -> bool:
            """Return whether this trade is already closed.
            """
            return self.exit_price > 0.

        def is_long(self) -> bool:
            """Return whether this trade is a LONG position.
            """
            return self.side == DayTrader.Position.LONG

        def is_short(self) -> bool:
            """Return whether this trade is a SHORT position.
            """
            return self.side == DayTrader.Position.SHORT

        def __str__(self) -> str:
            """Convert the trade details into a human readable string.
            """
            T = 'LONG' if self.is_long() else 'SHORT'
            return f'{T} ${self.investment:.2f}({self.num_shares}) -> ENTER:${self.enter_price:.2f}, EXIT:${self.exit_price:.2f}, PNL:${self.pnl:.2f}'

    def __init__(self, df: pd.DataFrame, features: list[str], balance: int = 10_000, verbose=True):
        """DayTrading evironment.
        """
        super().__init__(name='DayTrader', env_type=Env.DISCRETE, action_space=np.array(
            [DayTrader.NONE, DayTrader.BUY, DayTrader.SELL]), observation_space=df[features].to_numpy(dtype=np.float32))
        self.df: pd.DataFrame = df
        self.closing_prices: np.array[float] = self.df['Close'].to_numpy(
            dtype=np.float32)
        self.balance: float = balance
        self.verbose: bool = verbose
        self.reset()

    def reset(self, seed: int = None) -> tuple[State, any]:
        self.current_balance: float = self.balance
        self.last_balance: float = self.current_balance
        self.position_size: float = 2500.
        self.trade_history: list[DayTrader.Position] = []
        self.episode_reward: float = 0.
        self.min_profit: float = None
        self.expiration: int = None
        self.active_trade: DayTrader.Position = None
        return super().reset(seed)

    def step(self, action: Action) -> tuple[State, float, bool, bool, any]:
        """Perform a action in this environment and compute the reward, for the MDP. This function 
        returns the next state, the reward, if simulation is done or truncated. Note that to stay compatible
        with the gym environments we also return 'info' as last parameter and with None value in the tuple,
        like so: next_state, reward, done, truncated, info = env.step(action)
        """
        close = self.closing_prices[self.index]
        self.index += 1
        done = self.index >= self.max_steps or self.current_balance < 0.
        if not done:
            if action == DayTrader.BUY:
                # BUY action: Sell all open SHORT trades, and BUY into a LONG trade
                if self.active_trade and self.active_trade.is_short():
                    _, balance_change = self.active_trade.exit(close)
                    self.current_balance += balance_change
                    self.active_trade = None

                if self.active_trade is None:
                    self.active_trade = DayTrader.Position(
                        DayTrader.Position.LONG)
                    self.trade_history.append(self.active_trade)
                    self.current_balance -= self.active_trade.enter(
                        close, self.position_size, self.expiration)

            elif action == DayTrader.SELL:
                # SELL action: Sell all open LONG trades, and BUY into a SHORT trade
                if self.active_trade and self.active_trade.is_long():
                    _, balance_change = self.active_trade.exit(close)
                    self.current_balance += balance_change
                    self.active_trade = None

                if self.active_trade is None:
                    self.active_trade = DayTrader.Position(
                        DayTrader.Position.SHORT)
                    self.trade_history.append(self.active_trade)
                    self.current_balance -= self.active_trade.enter(
                        close, self.position_size, self.expiration)
                    
            reward = (self.current_balance - self.last_balance)/self.balance
            self.last_balance = self.current_balance
        else:
            print(
                f'Done after day #{self.index}: '
                f'Balance: ${self.current_balance:.2f}',
                f'Trades: {len(self.trade_history)}')
            reward = 0.

        self.episode_reward += reward
        # Return (state, reward, done)
        next_state = self.observation_space[self.index]
        return next_state, reward, done, False, None


class TrendFollow(Env):
    NONE: Action = 0
    BUY: Action = 1
    SELL: Action = 2

    def __init__(self, df, features):
        super().__init__(name='TrendFollow', env_type=Env.DISCRETE, action_space=np.array([
            TrendFollow.NONE, TrendFollow.BUY, TrendFollow.SELL]), observation_space=df[features].to_numpy(dtype=np.float32))

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
        if action == TrendFollow.BUY:
            self.invested = 1.
        elif action == TrendFollow.SELL:
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
        reward = self.rewards[self.index] * self.invested
        done = self.index >= self.max_steps
        return next_state, reward, done, False, None


class PredictUpDown(Env):
    UP: Action = 0
    DOWN: Action = 1

    def __init__(self, df, features):
        super().__init__(name='PredictUpDown', env_type=Env.DISCRETE, action_space=np.array([
            PredictUpDown.UP, PredictUpDown.DOWN]), observation_space=df[features].to_numpy(dtype=np.float32))
        self.df = df
        self.features = features
        self.returns = self.df['LogReturn'].to_numpy(dtype=np.float32)
        self.reset()

    def reset(self, seed: int = None) -> tuple[State, any]:
        return super().reset(seed)

    def step(self, action: Action) -> tuple[State, float, bool, bool, any]:
        """Perform a action in this environment and compute the reward, for the MDP. This function 
        returns the next state, the reward, if simulation is done or truncated. Note that to stay compatible
        with the gym environments we also return 'info' as last parameter and with None value in the tuple,
        like so: next_state, reward, done, truncated, info = env.step(action)
        """
        self.index += 1
        reward = 0
        if action == PredictUpDown.UP:
            reward += 1 if self.returns[self.index] > 0 else -1
        elif action == PredictUpDown.DOWN:
            reward += 1 if self.returns[self.index] <= 0 else -1
        next_state = self.observation_space[self.index]
        done = self.index >= self.max_steps
        return next_state, reward, done, False, None

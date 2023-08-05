from portcullis.env import Env, State, Action
import pandas as pd
import numpy as np
import math


class DayTrader(Env):
    # Action Types
    NONE: Action = 0
    LONG: Action = 1
    SHORT: Action = 2
    CLOSE: Action = 3

    class Trade():
        """Class handling trading functionalty.
        """

        def __init__(self, action: int, enter_price: float):
            self.action = action
            assert self.action == DayTrader.LONG or DayTrader.SHORT
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
            if self.action == DayTrader.LONG:
                self.pnl = (self.exit_price - self.enter_price) * \
                    self.num_shares
            elif self.action == DayTrader.SHORT:
                self.pnl = (self.enter_price - self.exit_price) * \
                    self.num_shares
            return self.pnl

        def is_closed(self) -> bool:
            return self.exit_price > 0.

        def __str__(self) -> str:
            T = 'LONG' if self.action == DayTrader.LONG else 'SHORT'
            return f'{T}[{self.num_shares}]: PNL ${self.pnl:.2f}'

    def __init__(self, df: pd.DataFrame, balance: int = 10_000, samples_per_day=1, verbose=True):
        """DayTrading evironment.
        """
        super().__init__(name='DayTrader', env_type=Env.DISCRETE, action_space=[
            DayTrader.NONE, DayTrader.LONG, DayTrader.SHORT, DayTrader.CLOSE], observation_space=df.to_numpy())
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

    def bankrupt(self) -> tuple[float, bool]:
        if self.verbose:
            print('Bot went bankrupt on day #', self.index)
        return -1., True

    def step(self, action: Action) -> tuple[State, float, bool, bool, any]:
        """Perform a action in this environment and compute the reward, for the MDP. This function 
        returns the next state, the reward, if simulation is done or truncated. Note that to stay compatible
        with the gym environments we also return 'info' as last parameter and with None value in the tuple,
        like so: next_state, reward, done, truncated, info = env.step(action)
        """
        self.index += 1
        done, trunc = False, False
        reward = 0.
        if action > DayTrader.NONE:
            close = self.closing_prices[self.index]
            if action == DayTrader.LONG or action == DayTrader.SHORT:
                if self.active_trade is None:
                    self.active_trade = DayTrader.Trade(
                        action, enter_price=close)
                    self.trades.append(self.active_trade)
                    if not self.active_trade.enter(
                        self.current_balance):
                        reward, done = self.bankrupt()
            elif action == DayTrader.CLOSE:
                if self.active_trade is not None:
                    pnl = self.active_trade.exit(
                        close)
                    self.current_balance += pnl
                    if self.current_balance <= 0.:
                        reward, done = self.bankrupt()
                    else:
                        reward = pnl / self.balance
                    if self.verbose:
                        print(
                            f'#{self.index + 1}: {self.active_trade} => ${self.current_balance:.2f}'
                        )
                    self.active_trade = None

        # Return (state, reward, done)
        self.episode_reward += reward
        next_state = self.observation_space[self.index]
        done = done or self.index >= self.max_steps
        if done:
            print(
                f'Done after day #{self.index}: Trades taken:{len(self.trades)}, Balance: ${self.current_balance:.2f}')
        return next_state, reward, done, trunc, None

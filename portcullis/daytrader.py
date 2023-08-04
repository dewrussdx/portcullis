from portcullis.env import Env, State, Action
import pandas as pd
import numpy as np


class DayTrader(Env):
    # Action Types
    NONE: Action = 0
    LONG: Action = 1
    SHORT: Action = 2
    CLOSE: Action = 3

    def __init__(self, df: pd.DataFrame, balance: int = 10_000, samples_per_day=1):
        """DayTrading evironment.
        """
        super().__init__(name='DayTrader', env_type=Env.DISCRETE, action_space=[
            DayTrader.NONE, DayTrader.LONG, DayTrader.SHORT, DayTrader.CLOSE], observation_space=df.to_numpy())
        self.df = df
        self.closing_prices = self.df['Close'].to_numpy(dtype=np.float32)
        self.starting_balance = balance
        self.current_balance = self.starting_balance
        self.samples_per_day = samples_per_day
        self.samples_remaining = self.samples_per_day
        self.max_invest = self.current_balance / 2
        self.reset()

    def reset(self, seed: int = None) -> tuple[State, any]:
        self.trade_type = DayTrader.NONE
        self.enter_price = 0.
        self.num_shares = 0
        return super().reset(seed)

    def step(self, action: Action) -> tuple[State, float, bool, bool, any]:
        """Perform a action in this environment and compute the reward, for the MDP. This function 
        returns the next state, the reward, if simulation is done or truncated. Note that to stay compatible
        with the gym environments we also return 'info' as last parameter and with None value in the tuple,
        like so: next_state, reward, done, truncated, info = env.step(action)
        """
        done = False
        self.index += 1
        assert self.index <= self.max_steps
        # Calculate reward for the step
        reward = 0.
        if action == DayTrader.LONG or action == DayTrader.SHORT:
            if self.num_shares > 0 or self.samples_remaining <= 0:
                reward += -1.  # Already invested, or too late to invest, don't do that again
            else:
                budget = min(self.max_invest, self.current_balance)
                closing_price = self.closing_prices[self.index]
                num_shares = int(budget / closing_price)
                if num_shares <= 0:
                    reward += -10.  # We ran out of money, punishable by death
                    done = True
                else:
                    self.num_shares = num_shares
                    self.current_balance -= self.num_shares * closing_price
                    assert self.current_balance >= 0.
                    self.trade_type = action
                    self.enter_price = closing_price
                    print('#', self.index, ':', 'LONG' if action == DayTrader.LONG else 'SHORT', self.num_shares, 'shares for $',
                          self.num_shares * closing_price, ', at $', closing_price, 'resulting in a balance of $', self.current_balance)
        elif action == DayTrader.CLOSE:
            if self.num_shares <= 0:
                reward += -1  # That was kinda dumb: Not invested, don't do that again
            else:
                assert self.trade_type != DayTrader.NONE
                closing_price = self.closing_prices[self.index]
                log_return = np.log(closing_price / self.enter_price)
                if self.trade_type == DayTrader.LONG:
                    self.current_balance += self.num_shares * closing_price
                    reward += log_return
                elif self.trade_type == DayTrader.SHORT:
                    self.current_balance += self.num_shares * \
                        (self.enter_price + (self.enter_price - closing_price))
                    if self.current_balance <= 0.:
                        reward += -10.  # Bankrupt
                        done = True
                    else:
                        reward += -log_return
                print('#', self.index, ':', 'CLOSE', self.num_shares, 'LONG' if self.trade_type == DayTrader.LONG else 'SHORT', 'shares for $',
                      self.num_shares * closing_price, ', at $', closing_price, 'resulting in a balance of $', self.current_balance)
                self.num_shares = 0
                self.trade_type = DayTrader.NONE
                self.enter_price = 0.
        else:  # DayTrader.NONE
            if self.num_shares <= 0:
                reward += -0.1  # Discourage idling, but not the end of the world

        self.samples_remaining -= 1
        if self.samples_remaining <= 0:
            self.samples_remaining = self.samples_per_day

        # Return (state, reward, done)
        next_state = self.observation_space[self.index]
        done = done or self.index >= self.max_steps
        return next_state, reward, done, False, None

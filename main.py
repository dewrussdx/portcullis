import pandas as pd
import numpy as np
import argparse
from portcullis.sim import Sim
from portcullis.replay import Mem
from portcullis.agent import DQN, TD3
from portcullis.env import Env
from portcullis.traders import DayTrader, FollowSMA, PredictSPY
from portcullis.portfolio import Portfolio
from portcullis.ta import SMA, EWMA
import gymnasium as gym

DEFAULT_RNG_SEED = 2170596287


def investment_portfolio_sample():
    Portfolio(['GOOG', 'SBUX', 'KSS', 'NEM']).export_timeseries_to_csv(
        csv='portfolio.csv',
        start='2014-01-02',
        end='2014-07-01')
    start = '2018-07-01'
    interval = '1d'
    # top10_sp500 = Portfolio.filter_top_return_symbols(
    #    Portfolio.fetch_sp500_symbols(), 10, start=start, interval=interval)
    top10_sp500 = ['ENPH', 'TSLA', 'MRNA', 'GEHC',
                   'CEG', 'AMD', 'SEDG', 'NVDA', 'CARR', 'DXCM']
    print(top10_sp500)
    sp500 = Portfolio(top10_sp500)
    if sp500.optimize(bounds=(0, None), risk_free_rate=0.03/252, start=start, interval=interval):
        print(sp500)


def swingtrading_portfolio_sample():
    portfolio = Portfolio([
        'SMH', 'XBI', 'META'
    ])
    asset = portfolio['META'].asset
    min_b, max_b = asset.get_swing_bounds(
        start='2018-07-01', interval='1wk') or (0, 0)
    print(asset)
    print(' Min: %.2f %%' % (min_b * 100.0))
    print('Mean: %.2f %%' % (max_b * 100.0))


def _followSMA(_args) -> Env:
    df = pd.read_csv('SPY.csv', index_col='Date', parse_dates=True)
    df['FastSMA'] = df['Close'].rolling(16).mean()
    df['SlowSMA'] = df['Close'].rolling(33).mean()
    df['LogReturn'] = np.log(df['Close']).diff()
    df.dropna(axis=0, how='any', inplace=True)
    env = FollowSMA(df=df.iloc[:-1000], features=['FastSMA', 'SlowSMA'])
    return env


def _predictSPY(_args) -> Env:
    df = pd.read_csv('sp500_closefull.csv', index_col=0, parse_dates=True)
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='any', inplace=True)
    df_returns = pd.DataFrame()
    features = ['AAPL', 'MSFT', 'AMZN']
    for name in features:
        df_returns[name] = np.log(df[name]).diff()
    df_returns['SPY'] = np.log(df['SPY']).diff()
    df_returns.dropna(axis=0, how='any', inplace=True)
    env = PredictSPY(df=df_returns.iloc[:-1000], features=features)
    return env


def _daytrader(args) -> Env:
    df = Env.yf_download('AAPL', start='2015-01-01',
                         features=['Close'], logret_col=None, ta=[])

    train, _ = Env.split_data(df, test_ratio=0.25)
    env = DayTrader(train, balance=10_000, verbose=args.verbose)
    return env


def create_trading_sim(args):
    env = _predictSPY(args)

    agent = None
    if args.algo == 'DQN':
        agent = DQN(env, mem=Mem(args.replaybuffer_size), hdims=(512, 256), lr=args.lr,
                    gamma=args.gamma, eps=args.eps, eps_min=args.eps_min, eps_decay=args.eps_decay,
                    tau=args.tau, name=f'DayTrader_DQNAgent')
    return agent


def create_gym_sim(args: list[any], render_mode='human') -> any:
    # 'CartPole-v1', 'LunarLander-v2', 'MountainCar-v0'
    # 'MountainCarContinuous-v0', 'HalfCheetah-v2'
    #
    # https://gymnasium.farama.org/
    env = gym.make(args.env, render_mode=render_mode)
    agent = None
    if args.algo == 'DQN':
        agent = DQN(env, mem=Mem(args.replaybuffer_size), hdims=(512, 256), lr=args.lr,
                    gamma=args.gamma, eps=args.eps, eps_min=args.eps_min, eps_decay=args.eps_decay,
                    tau=args.tau, name=f'{args.env}_DQNAgent')
    return agent


def main():
    parser = argparse.ArgumentParser()
    # Interactive mode
    parser.add_argument('--interactive', default=True, action='store_true')
    # Environment name (gym or native)
    # parser.add_argument('--env', default='DayTrader')
    parser.add_argument('--env', default='CartPole-v1')
    # Algo name (DQN, TD3, DDPG or DDPGplus)
    parser.add_argument('--algo', type=str.upper, default='DQN',
                        choices=['DQN', 'DQN_LAP', 'TD3', 'TD3_LAP'])
    # Number of episodes
    parser.add_argument('--num_episodes', default=1_000,
                        type=int)
    # Replaybuffer size
    parser.add_argument('--replaybuffer_size',
                        default=10_000, type=int)
    # Learning rate
    parser.add_argument('--lr',
                        default=1e-4, type=float)
    # Inference or training mode
    parser.add_argument('--mode', type=str.lower,
                        default='train', choices=['train', 'eval'])
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument('--seed', default=DEFAULT_RNG_SEED, type=int)
    # Ignore episode truncation flag
    parser.add_argument('--ignore_trunc', default=False, action='store_true')
    # Epsilon Initial
    parser.add_argument('--eps', default=1.0, type=float)
    # Epsilon Minimum
    parser.add_argument('--eps_min', default=0.005, type=float)
    # Epsilon Decay
    parser.add_argument('--eps_decay', default=0.9995, type=float)

    # Time steps initial random policy is used
    parser.add_argument('--start_timesteps', default=1_000, type=int)  # 25e3
    # How often (time steps) we evaluate
    parser.add_argument('--eval_freq', default=5e3, type=int)
    # Max time steps to run environment
    parser.add_argument('--max_timesteps', default=1e6, type=int)
    # Std of Gaussian exploration noise
    parser.add_argument('--expl_noise', default=0.1, type=float)
    # Batch size for both actor and critic
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--gamma', default=0.99,
                        type=float)        # Discount factor
    # Target network update rate
    parser.add_argument('--tau', default=0.005, type=float)
    # Noise added to target policy during critic update
    parser.add_argument('--policy_noise', default=0.2)
    # Range to clip target policy noise
    parser.add_argument('--noise_clip', default=0.5)
    # Frequency of delayed policy updates
    parser.add_argument('--policy_freq', default=2, type=int)
    # Verbose log output
    parser.add_argument('--verbose', default=False, action='store_true')
    # Save enabled
    parser.add_argument('--save', default=False, action='store_true')
    # Load enabled
    parser.add_argument('--load', default=False, action='store_true')
    # path for load/save (None=default path)
    parser.add_argument('--path', default=None)
    # path for load/save (None=default path)

    args = parser.parse_args()
    print(args)

    if args.env == 'DayTrader':
        agent = create_trading_sim(args)
    else:
        agent = create_gym_sim(args)
    assert agent is not None

    # ---------------
    # !!!HACK!!!
    # Run different simulation driver for environment type
    # FIXME: Should be handled by just one driver
    _, env_type, _, _, _ = Env.get_env_spec(agent.env)
    if env_type == Env.DISCRETE:
        Sim(agent).run(args)
        # Sim(agent).run_discrete_lap(agent.env, args)
    else:
        assert env_type == Env.CONTINUOUS
        Sim(agent).run_continuous(agent.env, args)
    # ---------------


if __name__ == "__main__":
    main()

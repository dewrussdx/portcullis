import os
import json
import argparse
import pandas as pd
import numpy as np
import gymnasium as gym
from portcullis.sim import Sim
from portcullis.replay import ReplayBuffer
from portcullis.agent import DQN
from portcullis.env import Env
from portcullis.traders import DayTrader, TrendFollow, PredictSPY, PredictUpDown
from portcullis.portfolio import Portfolio
from portcullis.ta import EWMA

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


def _trendFollow(_args, lookback: int = 7) -> Env:
    features = ['Close']
    df = Env.yf_download('AMZN', start='2015-01-01',
                         features=features, logret_col='Close', ta=[EWMA(20), EWMA(8)])
    train, _ = Env.split_data(df, test_ratio=0.2)
    env = TrendFollow(train, features)
    return env


def _predictSPY(_args) -> Env:
    raw = pd.read_csv('sp500_closefull.csv', index_col=0, parse_dates=True)
    raw.dropna(axis=0, how='all', inplace=True)
    raw.dropna(axis=1, how='any', inplace=True)
    df = pd.DataFrame()
    features = ['AAPL', 'MSFT', 'AMZN']
    for name in features:
        df[name] = np.log(raw[name]).diff()
    df['SPY'] = np.log(raw['SPY']).diff()
    df.dropna(axis=0, how='any', inplace=True)
    train, _ = Env.split_data(df, test_ratio=0.2)
    env = PredictSPY(train, features)
    return env


def _predictUpDown(_args) -> Env:
    features = ['Close', 'EWMA_8', 'EWMA_20']
    df = Env.yf_download('AAPL', start='2015-01-01',
                         features=features, logret_col='Close', ta=[EWMA(8), EWMA(20)])
    train, _ = Env.split_data(df, test_ratio=0.25)
    env = PredictUpDown(train, features)
    return env


def _daytrader(args) -> Env:
    df = Env.yf_download('AAPL', start='2015-01-01',
                         features=['Close'], logret_col=None, ta=[])

    train, _ = Env.split_data(df, test_ratio=0.25)
    env = DayTrader(train, balance=10_000, verbose=args.verbose)
    return env


def load_config(name: str = None) -> dict[str, any]:
    path = f'./configs/{name.lower()}.json'
    config = None
    if os.path.exists(path):
        print(f'Loading configuration from {path}...')
        with open(path, 'r') as handle:
            config = json.loads(handle.read())
            print(config)
    return config


def create_trading_sim(args, dispatcher):
    env = dispatcher[args.env.lower()](args)
    print('Trading environment:', env.name)
    _, _, _, state_dim, _ = Env.get_env_spec(env)
    replay_buffer = ReplayBuffer(
        state_dim=state_dim, capacity=args.mem_size, batch_size=args.batch_size, is_prioritized=(args.mem_type == 'LAP'))
    agent = None
    if args.algo == 'DQN':
        agent = DQN(env, mem=replay_buffer, hdims=(512, 256), lr=args.lr, gamma=args.gamma, eps=args.eps, eps_min=args.eps_min,
                    eps_decay=args.eps_decay, tau=args.tau, name=f'{env.name}_DQNAgent')
    return agent


def create_gym_sim(args: list[any], render_mode='human') -> any:
    # https://gymnasium.farama.org/
    print('Gym environment:', args.env)
    env = gym.make(args.env, render_mode=render_mode)
    _, _, _, state_dim, _ = Env.get_env_spec(env)
    replay_buffer = ReplayBuffer(
        state_dim=state_dim, capacity=args.mem_size, batch_size=args.batch_size, is_prioritized=(args.mem_type == 'LAP'))
    agent = None
    if args.algo == 'DQN':
        agent = DQN(env, mem=replay_buffer, hdims=(512, 256), lr=args.lr, gamma=args.gamma, eps=args.eps, eps_min=args.eps_min,
                    eps_decay=args.eps_decay, tau=args.tau, name=f'{args.env}_DQNAgent')
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
                        choices=['DQN', 'TD3'])
    # Number of episodes
    parser.add_argument('--num_episodes', default=500,
                        type=int)
    # Memory size
    parser.add_argument('--mem_size',
                        default=10_000, type=int)
    # Memory type
    parser.add_argument('--mem_type', type=str.upper,
                        default="LAP", choices=['LAP', 'UNI'])
    # Memory prefill
    parser.add_argument('--mem_prefill', default=256, type=int)
    # Learning rate
    parser.add_argument('--lr',
                        default=1e-4, type=float)
    # Inference or training mode
    parser.add_argument('--mode', type=str.lower,
                        default='train', choices=['train', 'eval'])
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument('--seed', default=DEFAULT_RNG_SEED, type=int)
    # Epsilon Initial
    parser.add_argument('--eps', default=0.9, type=float)
    # Epsilon Minimum
    parser.add_argument('--eps_min', default=0.05, type=float)
    # Epsilon Decay
    parser.add_argument('--eps_decay', default=1e3, type=float)
    # How often (time steps) we evaluate
    parser.add_argument('--eval_freq', default=5e3, type=int)
    # Std of Gaussian exploration noise
    parser.add_argument('--expl_noise', default=0.1, type=float)
    # Batch size for both actor and critic
    parser.add_argument('--batch_size', default=64, type=int)
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
    # Config file path (None=default path)
    parser.add_argument('--config', default=None)
    # Save enabled
    parser.add_argument('--save', default=False, action='store_true')
    # Load enabled
    parser.add_argument('--load', default=False, action='store_true')
    # path for load/save (None=default path)
    parser.add_argument('--path', default=None)

    # Load config file if available, override with commandline params
    args = parser.parse_args()
    config = load_config(args.env.lower())
    if config is not None:
        parser.set_defaults(**config['hyperparams'])
        del config['hyperparams']
        parser.set_defaults(**config)
    args = parser.parse_args()
    print(args)

    # Native/Trader simulation dispatcher
    trader_dispatcher = {
        'predictupdown': _predictUpDown,
        'predictspy': _predictSPY,
        'trendfollow': _trendFollow,
        'daytrader': _daytrader,
    }
    if args.env.lower() in trader_dispatcher.keys():
        agent = create_trading_sim(args, trader_dispatcher)
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

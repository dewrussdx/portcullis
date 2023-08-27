import os
import json
import argparse
import pandas as pd
import numpy as np
import gymnasium as gym
from portcullis.sim import Sim
from portcullis.replay import ReplayBuffer
from portcullis.agent import Agent, DQN
from portcullis.env import Env
from portcullis.traders import DayTrader, TrendFollow, PredictSPY, PredictUpDown
from portcullis.ta import EWMA
from portcullis.tictactoe import TicTacToe
DEFAULT_RNG_SEED = 2170596287


class EnvFactory:
    @staticmethod
    def create_trend_follow(args) -> Env:
        features = ['Close']
        df = Env.yf_download('AMZN', start='2015-01-01',
                             features=features, logret_col='Close', ta=[EWMA(20), EWMA(8)])
        train, test = Env.split_data(df, test_ratio=0.2)
        data = train if args.mode == 'train' else test
        env = TrendFollow(data, features)
        return env

    @staticmethod
    def create_predict_spy(args) -> Env:
        raw = pd.read_csv('sp500_closefull.csv', index_col=0, parse_dates=True)
        raw.dropna(axis=0, how='all', inplace=True)
        raw.dropna(axis=1, how='any', inplace=True)
        df = pd.DataFrame()
        features = ['AAPL', 'MSFT', 'AMZN']
        for name in features:
            df[name] = np.log(raw[name]).diff()
        df['SPY'] = np.log(raw['SPY']).diff()
        df.dropna(axis=0, how='any', inplace=True)
        train, test = Env.split_data(df, test_ratio=0.2)
        data = train if args.mode == 'train' else test
        env = PredictSPY(data, features)
        return env

    @staticmethod
    def create_predict_up_down(args) -> Env:
        features = ['Close', 'EWMA_8', 'EWMA_20']
        df = Env.yf_download('AAPL', start='2015-01-01',
                             features=features, logret_col='Close', ta=[EWMA(8), EWMA(20)])
        train, test = Env.split_data(df, test_ratio=0.25)
        data = train if args.mode == 'train' else test
        env = PredictUpDown(data, features)
        return env

    @staticmethod
    def create_daytrader(args) -> Env:
        features = ['Close']
        df = Env.yf_download('AAPL', start='2015-01-01',
                             features=features, logret_col='Close', ta=[])
        train, test = Env.split_data(df, test_ratio=0.25)
        data = train if args.mode == 'train' else test
        env = DayTrader(df=data, features=features,
                        balance=10_000, verbose=args.verbose)
        return env

    @staticmethod
    def get_trader_factory(name: str) -> any:
        registry = {
            'predictupdown': EnvFactory.create_predict_up_down,
            'predictspy': EnvFactory.create_predict_spy,
            'trendfollow': EnvFactory.create_trend_follow,
            'daytrader': EnvFactory.create_daytrader,
        }
        return registry.get(name.lower(), None)

    @staticmethod
    def create_trading_sim(args: list[any]) -> Agent:
        factory = EnvFactory.get_trader_factory(args.env)
        if not factory:
            return None
        env = factory(args)
        print('Trading environment:', env.name)
        _, _, _, state_dim, _ = Env.get_env_spec(env)
        replay_buffer = ReplayBuffer(
            state_dim=state_dim, capacity=args.mem_size, batch_size=args.batch_size, is_prioritized=(args.mem_type == 'LAP'))
        agent = None
        if args.algo == 'DQN':
            agent = DQN(env, mem=replay_buffer, hdims=(512, 256), lr=args.lr, gamma=args.gamma, eps=args.eps, eps_min=args.eps_min,
                        eps_decay=args.eps_decay, tau=args.tau, name=f'{env.name}_DQNAgent')
        return agent

    @staticmethod
    def create_gym_sim(args: list[any], render_mode='human') -> Agent:
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

    @staticmethod
    def create_tictactoe(args: list[any]) -> Agent:
        print('TicTacToe Game:', args.env)
        env = TicTacToe()
        replay_buffer = ReplayBuffer(
            state_dim=pow(3, 9), capacity=args.mem_size, batch_size=args.batch_size, is_prioritized=(args.mem_type == 'LAP'))
        agent = None
        if args.algo == 'DQN':
            agent = DQN(env, mem=replay_buffer, hdims=(512, 256), lr=args.lr, gamma=args.gamma, eps=args.eps, eps_min=args.eps_min,
                        eps_decay=args.eps_decay, tau=args.tau, name=f'{args.env}_DQNAgent')
        return agent


def load_config(name: str = None) -> dict[str, any]:
    path = f'./configs/{name.lower()}.json'
    config = None
    if os.path.exists(path):
        print(f'Loading configuration from {path}...')
        with open(path, 'r') as handle:
            config = json.loads(handle.read())
            print(config)
    return config


def main():
    parser = argparse.ArgumentParser()
    # Verbose log output
    parser.add_argument('--verbose', default=False, action='store_true')
    # Inference or training mode
    parser.add_argument('--mode', type=str.lower,
                        default='train', choices=['train', 'eval'])
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument('--seed', default=DEFAULT_RNG_SEED, type=int)
    # Environment name (gym or native)
    parser.add_argument('--env', default='CartPole-v1')
    # Config file path (None=default path)
    parser.add_argument('--config', default=None)
    # Save enabled
    parser.add_argument('--save', default=False, action='store_true')
    # Load enabled
    parser.add_argument('--load', default=False, action='store_true')
    # path for load/save (None=default path)
    parser.add_argument('--path', default=None)
  # Algo name (DQN, TD3, DDPG or DDPGplus)
    parser.add_argument('--algo', type=str.upper, default='DQN',
                        choices=['DQN', 'TD3'])
    # Batch size for experience replay buffer learning
    parser.add_argument('--batch_size', default=64, type=int)
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
    parser.add_argument('--gamma', default=0.99,
                        type=float)        # Discount factor
    # Target network update rate
    parser.add_argument('--tau', default=0.005, type=float)
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
    # Noise added to target policy during critic update
    parser.add_argument('--policy_noise', default=0.2)
    # Range to clip target policy noise
    parser.add_argument('--noise_clip', default=0.5)
    # Frequency of delayed policy updates
    parser.add_argument('--policy_freq', default=2, type=int)

    # Load config file if available, override with commandline params
    args = parser.parse_args()
    config = load_config(args.env.lower())
    if config is not None:
        parser.set_defaults(**config['hyperparams'])
        del config['hyperparams']
        parser.set_defaults(**config)
    args = parser.parse_args()
    print(args)

    if args.env == 'TicTacToe':
        agent = EnvFactory.create_tictactoe(args)
    else:
        agent = EnvFactory.create_trading_sim(args)
        if agent is None:
            agent = EnvFactory.create_gym_sim(args)
    assert agent is not None

    # ---------------
    # !!!HACK!!!
    # Run different simulation driver for environment type
    # FIXME: Should be handled by just one driver
    _, env_type, _, _, _ = Env.get_env_spec(agent.env)
    if env_type == Env.DISCRETE:
        Sim(agent).run(args)
    else:
        assert env_type == Env.CONTINUOUS
        Sim(agent).run_continuous(agent.env, args)
    # ---------------


if __name__ == "__main__":
    main()

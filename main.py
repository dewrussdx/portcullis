import argparse
from portcullis.sim import Sim
from portcullis.mem import Mem
from portcullis.agent import DQN, DDPG, DDPGplus, TD3
from portcullis.env import Env
from portcullis.daytrader import DayTrader
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


def create_trading_sim(args):
    df = Env.yf_download('AAPL', logret_col=None, ta=[
                         EWMA(8), EWMA(20), SMA(15), SMA(45)])
    train, _ = Env.split_data(df, test_ratio=0.2)
    env = DayTrader(train, balance=10_000, verbose=args.verbose)
    agent = None
    if args.algo == 'DQN':
        agent = DQN(env, mem=Mem(args.replaybuffer_size), hdims=(256, 256), lr=args.lr,
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
        agent = DQN(env, mem=Mem(args.replaybuffer_size), hdims=(256, 256), lr=args.lr,
                    gamma=args.gamma, eps=args.eps, eps_min=args.eps_min, eps_decay=args.eps_decay,
                    tau=args.tau, name=f'{args.env}_DQNAgent')
    return agent


def main():
    parser = argparse.ArgumentParser()
    # Interactive mode
    parser.add_argument('--interactive', default=True, action='store_true')
    # Environment name (gym or native)
    parser.add_argument('--env', default='DayTrader')
    # Algo name (DQN, TD3, DDPG or DDPGplus)
    parser.add_argument('--algo', type=str.upper, default='DQN',
                        choices=['DQN', 'DDPG', 'DDPGplus', 'TD3'])
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
    parser.add_argument('--batch_size', default=128, type=int)
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
    else:
        assert env_type == Env.CONTINUOUS
        args.algo = 'DDPG'
        Sim(agent).run_continuous(agent.env, args)
    # ---------------


if __name__ == "__main__":
    main()

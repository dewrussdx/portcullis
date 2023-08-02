from portcullis.sim import Sim
from portcullis.mem import Mem
from portcullis.agent import DQNNAgent
from portcullis.nn import DQNN
from portcullis.env import Env, DaleTrader
from portcullis.portfolio import Portfolio
from portcullis.ta import SMA, EWMA


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


def create_native_sim():
    df = Env.yf_download('AAPL', ta=[EWMA(8), EWMA(20), SMA(15), SMA(45)])
    train, _ = Env.split_data(df, test_ratio=0.2)
    env = DaleTrader(train, balance=50_000)
    agent = DQNNAgent(env, mem=Mem(50_000), hdims=(512, 256), lr=1e-5,
                      gamma=0.99, eps=1.0, eps_min=0.01, eps_decay=0.999999,
                      tau=0.001, training=True)
    return agent


def create_gym_sim(name: str = 'CartPole-v1', render_mode='human') -> any:
    import gymnasium as gym
    env = gym.make(name, render_mode=render_mode)
    agent = DQNNAgent(env, mem=Mem(50_000), hdims=(512, 256), lr=1e-4,
                      gamma=0.99, eps=0.9, eps_min=0.01, eps_decay=0.9999,
                      tau=0.005, training=True)
    return agent


def main():
    # agent = create_gym_sim()
    agent = create_native_sim()
    sim = Sim(agent)
    sim.run(num_episodes=10_000, mem_samples=128, training=True) # EVAL: seed=42


if __name__ == "__main__":
    main()

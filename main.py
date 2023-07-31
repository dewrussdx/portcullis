
from portcullis.sim import Sim
from portcullis.mem import Mem
from portcullis.agent import DQNNAgent
from portcullis.nn import DQNN
from portcullis.env import Env
from portcullis.portfolio import Portfolio


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


def DRLTest():
    features = ['Close']
    train, _ = Env.split_data(Env.yf_download(
        'AAPL', features=features), test_ratio=0.1)
    Sim.seed_rng()
    env = Env(train, features=features)
    nn_p = DQNN(input_size=1, hidden_size=128, output_size=env.num_actions(),
                seed=Sim.DEFAULT_RNG_SEED, name='DQNN_V')
    nn_t = DQNN(input_size=1, hidden_size=128, output_size=env.num_actions(),
                seed=Sim.DEFAULT_RNG_SEED, name='DQNN_T')
    agent = DQNNAgent(env, nn_p=nn_p, nn_t=nn_t, mem=Mem(131072),
                      lr=0.001, gamma=0.95, eps=1.0, eps_min=0.001, eps_decay=0.995)
    sim = Sim(agent)
    avg_score = sim.run(num_episodes=10000, mem_samples=128)
    print('Average Score:', avg_score)


def main():
    # investment_portfolio_sample()
    # swingtrading_portfolio_sample()
    # Level().analyze_and_plot('SMH')
    DRLTest()


if __name__ == "__main__":
    main()

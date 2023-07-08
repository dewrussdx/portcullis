from portcullis.portfolio import Portfolio, Asset


def investment_portfolio_sample():
    portfolio = Portfolio([
        'GOOG', 'SBUX', 'KSS', 'NEM'
    ])

    weights = portfolio.optimize(risk_free_rate=0.03/252,
                                 start='2014-01-01', end='2014-07-01')
    if weights is not None:
        print(f'Sharp Ratio: {portfolio.get_sharp_ratio()}')
        print(f'Weights: {portfolio.get_weights()}')


def swingtrading_portfolio_sample():
    portfolio = Portfolio([
        'SMH', 'XBI', 'META'
    ])

    asset = portfolio['META'].asset
    min, max = asset.get_swing_bounds(
        start='2018-07-01', interval='1wk')
    print(asset)
    print(' Min: %.2f %%' % (min * 100.0))
    print('Mean: %.2f %%' % (max * 100.0))

    asset = portfolio['SMH'].asset
    min, max = asset.get_swing_bounds(
        start='2018-07-01', interval='1wk')
    print(asset)
    print(' Min: %.2f %%' % (min * 100.0))
    print('Mean: %.2f %%' % (max * 100.0))
 
    asset = portfolio['XBI'].asset
    min, max = asset.get_swing_bounds(
        start='2018-07-01', interval='1wk')
    print(asset)
    print(' Min: %.2f %%' % (min * 100.0))
    print('Mean: %.2f %%' % (max * 100.0))

def main():
    investment_portfolio_sample()
    swingtrading_portfolio_sample()


if __name__ == "__main__":
    main()

from portcullis.portfolio import Portfolio


def investment_portfolio_sample():
    start = '2018-07-01'
    interval = '1d'
    # top10_sp500 = Portfolio.filter_top_return_symbols(
    #    Portfolio.fetch_sp500_symbols(), 10, start=start, interval=interval)
    top10_sp500 = ['ENPH', 'TSLA', 'MRNA', 'GEHC',
                   'CEG', 'AMD', 'SEDG', 'NVDA', 'CARR', 'DXCM']
    print(top10_sp500)
    sp500 = Portfolio(top10_sp500)
    if sp500.optimize(risk_free_rate=0.03/252, start=start, interval=interval):
        print(f'Sharp Ratio: {sp500.get_sharp_ratio()}')
        print(f'Weights: {sp500.get_weights()}')


def swingtrading_portfolio_sample():
    portfolio = Portfolio([
        'SMH', 'XBI', 'META'
    ])

    asset = portfolio['META'].asset
    min, max = asset.get_swing_bounds(
        start='2018-07-01', interval='1wk') or (0, 0)
    print(asset)
    print(' Min: %.2f %%' % (min * 100.0))
    print('Mean: %.2f %%' % (max * 100.0))


def main():
    investment_portfolio_sample()
    # swingtrading_portfolio_sample()


if __name__ == "__main__":
    main()

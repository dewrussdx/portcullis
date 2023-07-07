from portcullis.portfolio import Portfolio, Asset


def main():
    portfolio = Portfolio([
        'GOOG', 'SBUX', 'KSS', 'NEM'
    ])

    # portfolio.export_timeseries_to_csv(
    #    'portfolio.csv', start='2014-01-01', end='2014-07-01')

    weights = portfolio.optimize(risk_free_rate=0.03/252,
                                 start='2014-01-01', end='2014-07-01')
    if weights is not None:
        print(f'Sharp Ratio: {portfolio.get_sharp_ratio()}')
        print(f'Weights: {portfolio.get_weights()}')


if __name__ == "__main__":
    main()

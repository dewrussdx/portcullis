
import argparse
from portcullis.portfolio import Portfolio


class PortfolioSamples:
    @staticmethod
    def investment_portfolio():
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

    @staticmethod
    def swingtrading_portfolio():
        portfolio = Portfolio([
            'SMH', 'XBI', 'META'
        ])
        asset = portfolio['META'].asset
        min_b, max_b = asset.get_swing_bounds(
            start='2018-07-01', interval='1wk') or (0, 0)
        print(asset)
        print(' Min: %.2f %%' % (min_b * 100.0))
        print('Mean: %.2f %%' % (max_b * 100.0))


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()


if __name__ == "__main__":
    main()

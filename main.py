from portcullis.portfolio import Portfolio


def main():
    portfolio = Portfolio([
        'MSFT', 'GOOG'
    ])

    msft = portfolio.get_stock('msft')
    print(msft.get_ticker().info)
    print(msft.get_timeseries(start='2023-01-01'))

    goog = portfolio.get_stock('goog')
    print(goog.get_ticker().info)
    print(goog.get_timeseries())


if __name__ == "__main__":
    main()

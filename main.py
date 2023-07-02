from portcullis.portfolio import Portfolio


def main():
    portfolio = Portfolio([
        'MSFT', 'GOOG'
    ])

    for key, value in portfolio.items():
        print(f'{key}: {value}')

    msft = portfolio['msft']
    print(msft.asset.get_ticker().info)
    print(msft.asset.get_timeseries(start='2023-01-01'))


if __name__ == "__main__":
    main()

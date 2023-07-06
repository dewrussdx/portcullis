from portcullis.portfolio import Portfolio


def main():
    portfolio = Portfolio([
        'MSFT', 'GOOG'
    ])

    for key, value in portfolio.items():
        print(f'{key}: {value}')

    mean_returns = portfolio.get_mean_returns(start='2023-01-01')
    print(mean_returns)
    
if __name__ == "__main__":
    main()

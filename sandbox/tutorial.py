
from scipy.optimize import linprog, minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def softmax(w):
    a = np.exp(w)
    return a / a.sum()


def sandbox1():
    np.random.seed(11111111)

    mean_return = 0.01 * np.random.randn(2)
    print(mean_return)

    rho = 0.01 * np.random.randn()
    print(rho)

    sigmas = np.exp(np.random.randn(2))
    print(sigmas)

    cov = np.diag(sigmas ** 2)
    print(cov)

    sigma12 = sigmas[0] * sigmas[1] * rho
    cov[0, 1] = sigma12
    cov[1, 0] = sigma12
    print(cov)

    N = 1000
    returns = np.zeros(N)
    risks = np.zeros(N)
    for i in range(N):
        # create random weights for the portfolio, that add up to 1 (using softmax)
        w = softmax(np.random.randn(2))
        # Calculate expected return, which is simply the mean return multiplied by the weight (vectorized to dot product)
        ret = mean_return.dot(w)
        # Calculate variance
        var = w.dot(cov).dot(w)
        # Calculate risk (square root of the variance)
        risk = np.sqrt(var)
        # Store the return and risk values
        returns[i] = ret
        risks[i] = risk

    for i in range(N):
        # Generate random number [0,1)
        x = np.random.random()
        # Generate weights (adding up to 1)
        w = np.array([x, 1-x])
        # Calculate expected return, which is simply the mean return multiplied by the weight (vectorized to dot product)
        ret = mean_return.dot(w)
        # Calculate variance
        var = w.dot(cov).dot(w)
        # Calculate risk (square root of the variance)
        risk = np.sqrt(var)
        returns[i] = ret
        risks[i] = risk

    for i in range(N):
        # Generate random number [-0.5,0.5)
        x = np.random.random() - 0.5
        # Generate weights (adding up to 1) allowing short selling (negative weights)
        w = np.array([x, 1-x])
        # Calculate expected return, which is simply the mean return multiplied by the weight (vectorized to dot product)
        ret = mean_return.dot(w)
        # Calculate variance
        var = w.dot(cov).dot(w)
        # Calculate risk (square root of the variance)
        risk = np.sqrt(var)
        returns[i] = ret
        risks[i] = risk

    print('--------------------------------')
    # 3 asset portfolio
    mean_return = 0.01 * np.random.randn(3)
    print(mean_return)

    sigmas = np.exp(np.random.randn(3))
    print(sigmas)

    rhos = 0.01 * np.random.randn(3)
    print(rhos)

    sigma01 = rhos[0] * sigmas[0] * sigmas[1]
    sigma02 = rhos[1] * sigmas[0] * sigmas[2]
    sigma12 = rhos[2] * sigmas[1] * sigmas[2]

    cov = np.array([
        [sigmas[0]**2, sigma01, sigma02],
        [sigma01, sigmas[1]**2, sigma12],
        [sigma02, sigma12, sigmas[2]**2]
    ])
    print(cov)

    for i in range(N):
        # Generate random number [-0.5,0.5)
        x1, x2 = np.random.random(2) - 0.5
        # Generate weights (adding up to 1) allowing short selling (negative weights)
        w = np.array([x1, x2, 1-x1-x2])
        # Shuffle weights for balanced values
        np.random.shuffle(w)
        # Calculate expected return, which is simply the mean return multiplied by the weight (vectorized to dot product)
        ret = mean_return.dot(w)
        # Calculate variance
        var = w.dot(cov).dot(w)
        # Calculate risk (square root of the variance)
        risk = np.sqrt(var)
        returns[i] = ret
        risks[i] = risk

    plt.scatter(risks, returns, alpha=0.1)
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.show()


def sandbox2():
    df = pd.read_csv('sp500sub.csv', index_col='Date', parse_dates=True)
    print(df.columns)
    print(df['Name'].unique())
    names = ['GOOG', 'SBUX', 'KSS', 'NEM']
    all_dates = df.index.unique().sort_values()
    print(len(all_dates))
    print(all_dates.get_loc('2014-01-02'))
    print(all_dates.get_loc('2014-06-30'))
    start = all_dates.get_loc('2014-01-02')
    end = all_dates.get_loc('2014-06-30')
    dates = all_dates[start:end+1]
    print(type(dates))
    print(dates)
    close_prices = pd.DataFrame(index=dates)
    for name in names:
        tmp = df.loc[dates]
        df_sym = tmp[tmp['Name'] == name]
        df_tmp = pd.DataFrame(
            data=df_sym['Close'].to_numpy(), index=df_sym.index, columns=[name])
        close_prices = close_prices.join(df_tmp)
    print(close_prices.head())
    num_na = close_prices.isna().sum().sum()
    print(f'num_na: {num_na}')
    close_prices.fillna(method='ffill', inplace=True)
    print(close_prices.isna().sum().sum())
    close_prices.fillna(method='bfill', inplace=True)
    print(close_prices.isna().sum().sum())
    returns = pd.DataFrame(index=dates[1:])
    for name in names:
        current_returns = close_prices[name].pct_change()
        returns[name] = current_returns.iloc[1:] * 100
    print(returns.head())
    mean_return = returns.mean()
    print(mean_return)
    cov = returns.cov()
    print(cov)
    cov_np = cov.to_numpy()
    print(cov_np)
    N = 10000
    D = len(mean_return)
    returns = np.zeros(N)
    risks = np.zeros(N)
    random_weights = []
    for i in range(N):
        rand_range = 1.0
        w = np.random.random(D)*rand_range - rand_range / \
            2  # with short-selling
        w[-1] = 1 - w[:-1].sum()
        np.random.shuffle(w)
        ret = mean_return.dot(w)
        var = w.dot(cov_np).dot(w)
        risk = np.sqrt(var)
        returns[i] = ret
        risks[i] = risk
        random_weights.append(w)

    single_asset_returns = np.zeros(D)
    single_asset_risks = np.zeros(D)
    for i in range(D):
        ret = mean_return[i]
        # covariance matrix contains the variance for asset i at index [i,i]
        risk = np.sqrt(cov_np[i, i])
        single_asset_returns[i] = ret
        single_asset_risks[i] = risk

    D = len(mean_return)
    print(f'D={D}')
    A_eq = np.ones((1, D))
    print(f'A_eq={A_eq}')
    b_eq = np.ones(1)
    print(f'b_eq={b_eq}')

    # Note: scipi linprog bounds are by default (0, None) unless otherwise specified!
    bounds = [(-0.5, None)] * D
    print(f'Bounds: {bounds}')

    # Run linear regression, by default this function minimizes
    res = linprog(mean_return, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    print(res)
    min_return = res.fun

    # Maximize
    res = linprog(-mean_return, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    print(res)
    max_return = -res.fun

    print(min_return, max_return)

    N = 100
    target_returns = np.linspace(min_return, max_return, num=N)
    print(target_returns)

    def get_portfolio_variance(weights):
        return weights.dot(cov).dot(weights)

    # For the scipy.minimize optimizer, the constraint function returns 0 if constraint has been met
    # > Optimize within target return value
    def target_return_constraint(weights, target):
        return weights.dot(mean_return) - target

    # For the scipy.minimize optimizer, the constraint function returns 0 if constraint has been met
    # > All weights need to add up to one
    def portfolio_constraint(weights):
        return weights.sum() - 1

    constraints = [
        {
            'type': 'eq',
            'fun': target_return_constraint,
            'args': [target_returns[0]],  # wlll be updated in loop
        },
        {
            'type': 'eq',
            'fun': portfolio_constraint,
        }
    ]
    x0 = np.ones(D)/D
    print(f'x0={x0}')
    # check if it works...
    res = minimize(
        fun=get_portfolio_variance,
        x0=np.ones(D) / D,  # Initial guess for the weights
        method='SLSQP',
        constraints=constraints,
    )
    print(res)

    bounds = [(-0.5, None)] * D
    print(f'Bounds: {bounds}')

    # Run again, this time bounding the weights...
    res = minimize(
        fun=get_portfolio_variance,
        x0=np.ones(D) / D,  # Initial guess for the weights
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
    )
    print(res)

    optimized_risks = []
    for target in target_returns:
        # Update argument for first constraint (target_return_constraint)
        constraints[0]['args'] = [target]
        # Run optimizer
        res = minimize(
            fun=get_portfolio_variance,
            x0=np.ones(D) / D,  # Initial guess for the weights
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
        )
        # Check optimizer result
        if res.status != 0:
            # Print optimizer result, something went wrong...
            print(res)
        else:
            # Store risk (standard deviation of variance)
            optimized_risks.append(np.sqrt(res.fun))

    # Compute the minimum variance portfolio
    # Run again, this time bounding the weights...
    res = minimize(
        fun=get_portfolio_variance,  # objective
        x0=np.ones(D) / D,  # Initial guess for the weights
        method='SLSQP',  # optimizing method
        constraints=[
            {
                'type': 'eq',
                'fun': portfolio_constraint,
            }
        ],
        bounds=bounds,
    )
    print(res)
    mv_risk = np.sqrt(res.fun)
    mv_weights = res.x
    mv_returns = mv_weights.dot(mean_return)

    # SHARP RATIO
    # => Basically: Return/Risk Ratio
    # => Definition: (Expected Return-Risk Free Rate) / (Standard Deviation)
    # Risk free rate are subtracted from asset's return rate

    # T-Bill Rate: https://fred.stlouisfed.org/series/TB3MS
    risk_free_rate = 5.16 / 252  # Annual to daily rate
    risk_free_rate = 0.03 / 252  # Tutorial values!

    # Compute negative sharp ratio
    # Note: Negative, since we want to maximize this value
    def neg_sharp_ratio(weights):
        expected_return = weights.dot(mean_return)
        # Risk (or Standard deviation) == sqrt(variance)
        std = np.sqrt(weights.dot(cov).dot(weights))
        return -(expected_return-risk_free_rate)/std

    res = minimize(
        fun=neg_sharp_ratio,  # objective
        x0=np.ones(D) / D,  # Initial guess for the weights
        method='SLSQP',  # optimizing method
        constraints=[
            {
                'type': 'eq',
                'fun': portfolio_constraint,
            }
        ],
        bounds=bounds,
    )
    print(res)
    best_sr, best_w = -res.fun, res.x
    print(f'best_sr={best_sr}, best_w={best_w}')
    # Let's experiment with monte-carlo simulation, trying to find the best solution
    mc_best_w = None
    mc_best_sr = float('-inf')
    for i, (risk, ret) in enumerate(zip(risks, returns)):
        sr = (ret-risk_free_rate)/risk
        if sr > mc_best_sr:
            mc_best_sr = sr
            mc_best_w = random_weights[i]
    print(f'mc_best_sr={mc_best_sr}, mc_best_w={mc_best_w}')

    # plot it
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.scatter(risks, returns, alpha=0.1)
    plt.plot(optimized_risks, target_returns, c='black')
    # found by optimization
    opt_risk = np.sqrt(best_w.dot(cov).dot(best_w))
    opt_ret = mean_return.dot(best_w)
    plt.scatter([opt_risk], [opt_ret], c='red')

    # found by monte carlo simulation
    #mc_risk = np.sqrt(mc_best_w.dot(cov).dot(mc_best_w))
    #mc_ret = mean_return.dot(mc_best_w)
    #plt.scatter([mc_risk], [mc_ret], c='pink')

    # Tangent line
    x1 = 0
    y1 = risk_free_rate
    x2 = opt_risk
    y2 = opt_ret
    plt.plot([x1, x2], [y1, y2], c='green')

    # Notes: 
    # 1. Portfolio with highest sharp ratio is also called Tangent portfolio
    # 2. Including risk free assets, and following the tangent line, better portfolios can be found

    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.show()


sandbox2()

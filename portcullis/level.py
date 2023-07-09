'''
Level Class
Functionality to compute support (bullish) and resistance (bearish) levels and areas
'''
from portcullis.util import today
from portcullis.asset import Asset

import pandas as pd
import numpy as np
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
import mplcursors


class Level:
    SUPPORT = 'Support'
    RESISTANCE = 'Resistance'

    def __init__(self):
        self.levels = []

    def plot_all(self, name, df):
        # Plotting the candlestick chart
        fig, ax = plt.subplots()
        candlestick_ohlc(ax, df.values, width=0.6,
                         colorup='green', colordown='red', alpha=0.8)
        ax.xaxis.set_tick_params(rotation=45)
        date_format = mpl_dates.DateFormatter('%d %b %Y')
        ax.xaxis.set_major_formatter(date_format)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('Stock Price of '+name)
        fig.autofmt_xdate()
        fig.tight_layout()

        # Plotting support and resistance levels
        for level in self.levels:
            plt.hlines(level[1], xmin=df['Date'][level[0]], xmax=max(
                df['Date']), colors='blue', linewidth=0.5)
            plt.text(df['Date'][level[0]], level[1], level[2],
                     ha='center', va='bottom', color='black')
            price = df['Close'][level[0]]
            volume = df['Volume'][level[0]]  # Added Volume
            plt.annotate(f'Price: {price:.2f}', (df['Date'][level[0]], level[1]), xytext=(5, -15),
                         textcoords='offset points', ha='right', color='black', size=8)  # Updated annotation for price
            plt.annotate(f'Volume: {volume:.0f}', (df['Date'][level[0]], level[1]), xytext=(5, -30),
                         textcoords='offset points', ha='right', color='black', size=8)  # Added annotation for volume

        # Plotting entry points
        entry_points = self.find_entry_points()
        for entry_point in entry_points:
            plt.scatter(df['Date'][entry_point], df['Low']
                        [entry_point], color='green', marker='o', s=100)
            plt.annotate('Time to Enter Trade', (df['Date'][entry_point], df['Low'][entry_point]), xytext=(15, -25),
                         textcoords='offset points', ha='left', color='white', size=10, backgroundcolor='purple', weight='bold',
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='green'))

        mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(
            f"Price: {sel.target[1]:.2f}"))  # Updated annotation
        plt.show()

    def analyze_and_plot(self, symbol):
        plt.rcParams['figure.figsize'] = [12, 7]
        plt.rc('font', size=14)
        df = self.get_stock(symbol)
        if df is None:
            return
        df['Date'] = pd.to_datetime(df.index)
        df['Date'] = df['Date'].apply(mpl_dates.date2num)
        df = df.loc[:, ['Date', 'Open', 'High', 'Low',
                        'Close', 'Volume']]  # Added Volume column
        self.compute_levels(df)
        self.plot_all(symbol, df)

    def find_breakout_candidates(self):
        # get the full stock list of S&P 500
        payload = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        stock_list = payload[0]['Symbol'].values.tolist()

        # find breakout candidates
        candidates = []
        for symbol in stock_list:
            df = self.get_stock(symbol)
            if df is not None:
                self.compute_levels(df)
                if (self.has_breakout(levels[-5:], df.iloc[-2], df.iloc[-1])):
                    candidates.append(symbol)
        print(candidates)
        return candidates

    def get_stock(self, symbol):
        return Asset(symbol).get_timeseries(interval='1d',
                                            start='2023-01-01', end=today())

    def is_far_from(self, value, df) -> bool:
        mean = np.mean(df['High'] - df['Low'])
        return np.sum([abs(value-level[1]) < mean for level in self.levels]) == 0

    # determine support/bullish fractal
    def is_support(self, df, i) -> bool:
        sample = df['Low']
        return sample[i] < sample[i-1] and sample[i] < sample[i+1] and sample[i+1] < sample[i+2] and sample[i-1] < sample[i-2]

    # determine resistance/bearish fractal
    def is_resistance(self, df, i) -> bool:
        sample = df['High']
        return sample[i] > sample[i-1] and sample[i] > sample[i+1] and sample[i+1] > sample[i+2] and sample[i-1] > sample[i-2]

    # Detect breakout
    def has_breakout(self, previous, last) -> bool:
        for _, value, _ in self.levels:
            lhs = previous['Open'] < value
            rhs = last['Open'] > value and last['Low'] > value
        return lhs and rhs

    # Find entry points
    def find_entry_points(self) -> list[int]:
        entry_points = []
        for i in range(1, len(self.levels)-1):
            curr_level = self.levels[i]
            prev_level = self.levels[i-1]
            next_level = self.levels[i+1]
            if curr_level[1] < prev_level[1] and curr_level[1] < next_level[1]:
                entry_points.append(curr_level[0])
        return entry_points

    def compute_levels(self, df) -> None:
        self.levels.clear()
        for i in range(2, df.shape[0]-2):
            if self.is_support(df, i):
                value = df['Low'][i]
                if self.is_far_from(value, df):
                    self.levels.append((i, value, Level.SUPPORT))
            elif self.is_resistance(df, i):
                value = df['High'][i]
                if self.is_far_from(value, df):
                    self.levels.append((i, value, Level.RESISTANCE))

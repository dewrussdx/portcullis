import pandas as pd


class TimeSeries:
    def __init__(self, symbols=None):
        self.full_df = pd.DataFrame()
        self.series = dict()
        self.add(symbols or [])

    def add(self, symbols):
        assert isinstance(symbols, list)
        for symbol in symbols:
            self.series[symbol.upper()] = dict()
        print(self.series)


class YFTimeSeries(TimeSeries):
    import yfinance as yf

    def download(self, **kwargs):
        kwargs["progress"] = False
        for key, _ in self.series.items():
            print(f"Downloading data for symbol '{key}'...")
            df = pd.DataFrame(YFTimeSeries.yf.download(key, **kwargs))
            df["Name"] = key
            # print(df)
            self.series[key] = df
            self.full_df = pd.concat([self.full_df, df], ignore_index=True)
        print(self.full_df)

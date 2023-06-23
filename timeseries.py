class TimeSeries:
    def __init__(self, symbols=None):
        self.series = {}
        self.add(symbols or [])

    def add(self, symbols):
        assert isinstance(symbols, list)
        for symbol in symbols:
            self.series[symbol.upper()] = {}


class YFTimeSeries(TimeSeries):
    import yfinance as yf

    def download(self, **kwargs):
        kwargs["progress"] = False
        for key, _ in self.series.items():
            print(key)
            data = YFTimeSeries.yf.download(key, **kwargs)
            data.to_csv(f"./data/{key}.csv")
            self.series[key] = data
            print(self.series[key])

from timeseries import YFTimeSeries

__version__ = "0.1"


def main():
    print(f"Portcullis Version {__version__}")
    ts = YFTimeSeries(["msft", "goog"])
    ts.download(start="2023-01-01")


if __name__ == "__main__":
    main()

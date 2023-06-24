from portcullis.timeseries import YFTimeSeries


def main():
    ts = YFTimeSeries(["msft", "goog"])
    ts.download(start="2023-01-01")


if __name__ == "__main__":
    main()

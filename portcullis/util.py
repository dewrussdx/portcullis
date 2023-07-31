'''
Utility functions
'''
import datetime
from collections import namedtuple
from datetime import datetime, timedelta
import pandas as pd


def today() -> datetime:
    return datetime.datetime.now()


def str2bool(s: str) -> bool:
    mapper = {
        '0': False,
        '1': True,
        'false': False,
        'true': True,
    }
    return mapper.get(s.lower(), False)

# Return S&P 500 symbols


def fetch_sp500_symbols() -> list[str]:
    df = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return df['Symbol'].tolist()

# Return whether day is a saturday


def is_saturday(day: datetime) -> bool:
    return day.weekday() == 5

# Return previous saturday from today


def prev_saturday(day: datetime) -> str:
    return day-timedelta((day.weekday()+2) % 7)

# Normalize datetime (set time component to 00:00:00)


def normalize_datetime(dt: datetime) -> datetime:
    return str2dt(dt2str(dt))

# String to datetime


def str2dt(s: str) -> datetime:
    return datetime.strptime(s, '%Y-%m-%d')


# Datetime to string
def dt2str(dt: datetime) -> str:
    return dt.strftime('%Y-%m-%d')


# Datetime to string (safe)
def sdt2str(dt: datetime) -> str:
    return dt2str(dt) if dt is not None else 'NA'

# Format number to 2 decimal places string


def flt2str(f: float) -> str:
    return '%.2f' % f

# Percentage string from fraction rounded to 2 decimal places


def frac2str(nom: float, denom: float) -> str:
    if abs(denom) > 0.000001:
        return flt2str((nom*100) / denom) + '%'
    return '0%'

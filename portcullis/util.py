'''
Utility functions
'''
import datetime


def today() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d')

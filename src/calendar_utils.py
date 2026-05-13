import pandas as pd
import math


def next_working_day(date):
    """
    move date to the next working day
    """
    date = pd.Timestamp(date)

    if date.weekday() >= 5:
        date = date + pd.offsets.BDay(1)

    return date


def add_working_days_floor(start_date, offset_days):
    """
    convert solver offset into planned start date
    floor since hourly breakdown not used currently
    """
    current = next_working_day(start_date)
    whole_days = int(math.floor(offset_days))

    return (current + pd.offsets.BDay(whole_days)).date()


def add_working_days_ceiling(start_date, offset_days):
    """
    convert solver offset into planned end date
    ceiling to show what day it'll be complete by
    """
    current = next_working_day(start_date)
    whole_days = int(math.ceil(offset_days))

    return (current + pd.offsets.BDay(whole_days)).date()
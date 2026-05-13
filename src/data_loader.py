import pandas as pd
import numpy as np
from pathlib import Path


def load_and_prepare_data():
    """
    load and prepare data for use in scheduler
    takes orders and cycle_times and merges the two
    capacity is not considered at this time
    """

    # set directory
    BASE_DIR = Path(__file__).resolve().parent.parent

    # load csv files
    orders_path = BASE_DIR / 'data' / 'orders.csv'
    cycle_times_path = BASE_DIR / 'data' / 'cycle_times.csv'
    capacity_path = BASE_DIR / 'data' / 'capacity.csv'

    # read orders, define dates, define descriptive columns
    orders = pd.read_csv(
        orders_path,
        parse_dates=['date', 'earlieststartdate'],
        dtype={
            'salesid': str,
            'itemid': str,
            'accountnum': str
        }
    )

    # read cycle_times
    cycle_times = pd.read_csv(
        cycle_times_path,
        dtype={
            'itemid': str
        }
    )

    # read capacity (empty right now)
    capacity = pd.read_csv(capacity_path)

    # only keep orders marked with production
    orders = orders[orders['prod'] == 1].copy()

    # sort orders for setting batch numbers
    orders = orders.sort_values(
        ['salesid', 'itemid', 'date', 'priority', 'qty'],
        ascending=[True, True, True, True, False]
    ).reset_index(drop=True)

    # set batch numbers
    orders['batch_number'] = (
        orders
        .groupby(['salesid', 'itemid'])
        .cumcount()
        .add(1)
    )

    # create new column batchid for solver
    orders['batchid'] = (
        orders['salesid'].astype(str)
        + '-'
        + orders['itemid'].astype(str)
        + '-B'
        + orders['batch_number'].astype(str).str.zfill(3)
    )

    # merge with cycle_times for batch/operations combinations
    df = orders.merge(cycle_times, on='itemid', how='left')

    # calculate operations times based on qty
    df['total_production_days'] = (
        df['qty'] * df['cycle_time_days']
    )

    # burn-in in sets of 4
    df.loc[df['sequence'] == 3, 'total_production_days'] = np.ceil(
        df.loc[df['sequence'] == 3, 'qty'] / 4
    )

    # fqc set to 1 day for all batch quantities
    df.loc[df['sequence'] == 4, 'total_production_days'] = 1

    # initial sort of values
    df = df.sort_values(
        [
            'priority',
            'qty',
            'date',
            'salesid',
            'itemid',
            'batch_number',
            'sequence'
        ],
        ascending=[
            True,
            False,
            True,
            True,
            True,
            True,
            True
        ]
    ).reset_index(drop=True)

    return df, capacity
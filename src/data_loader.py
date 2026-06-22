import pandas as pd
import numpy as np
from pathlib import Path


def load_and_prepare_data():
    """
    Load and prepare data for use in scheduler.

    This function:
    - loads order demand
    - separates prod = 1 and prod = 0 rows
    - creates batch IDs
    - expands production orders into operation rows using cycle_times
    - merges process requirements onto each operation
    - loads labor/resources, work centers, and resource eligibility tables
    """

    # set directory
    BASE_DIR = Path(__file__).resolve().parent.parent

    # load csv files
    orders_path = BASE_DIR / 'data' / 'orders.csv'
    cycle_times_path = BASE_DIR / 'data' / 'cycle_times.csv'
    processes_path = BASE_DIR / 'data' / 'processes.csv'
    resources_path = BASE_DIR / 'data' / 'resources.csv'
    workcenters_path = BASE_DIR / 'data' / 'workcenters.csv'
    resource_process_eligibility_path = BASE_DIR / 'data' / 'resource_process_eligibility.csv'

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

    # read process requirements
    processes = pd.read_csv(
        processes_path,
        dtype={
            'process': str
        }
    )

    # read labor resources
    resources = pd.read_csv(
        resources_path,
        dtype={
            'resourceid': str
        }
    )

    # read physical work centers
    workcenters = pd.read_csv(
        workcenters_path,
        dtype={
            'workcenterid': str,
            'process': str
        }
    )

    # read which resources can perform which processes
    resource_process_eligibility = pd.read_csv(
        resource_process_eligibility_path,
        dtype={
            'resourceid': str,
            'process': str
        }
    )

    # separate non production orders
    non_production_orders = orders[orders['prod'] == 0].copy()

    # only keep orders marked with production for the scheduler
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

    # add process requirements to each cycle time row
    cycle_times = cycle_times.merge(
        processes,
        on='process',
        how='left'
    )

    # merge with cycle_times for batch/operations combinations
    df = orders.merge(cycle_times, on='itemid', how='left')

    # validate that every scheduled item has cycle time rows
    if df['sequence'].isnull().any():
        missing_items = df[df['sequence'].isnull()]['itemid'].unique()
        raise ValueError(f'Missing cycle time rows for itemids: {missing_items}')

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

    return df, non_production_orders, resources, workcenters, resource_process_eligibility
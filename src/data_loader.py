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
    - loads workcenter capacity and labor capacity by process
    """

    # set directory
    BASE_DIR = Path(__file__).resolve().parent.parent

    # load csv files
    orders_path = BASE_DIR / 'data' / 'orders.csv'
    cycle_times_path = BASE_DIR / 'data' / 'cycle_times.csv'
    processes_path = BASE_DIR / 'data' / 'processes.csv'
    workcenters_path = BASE_DIR / 'data' / 'workcenters.csv'
    workcenter_eligibility_path = BASE_DIR / 'data' / 'workcenter_eligibility.csv'
    labor_capacity_path = BASE_DIR / 'data' / 'labor_capacity.csv'
    labor_eligibility_path = BASE_DIR / 'data' / 'labor_eligibility.csv'

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


    # read physical work centers
    workcenters = pd.read_csv(
        workcenters_path,
        dtype={
            'workcenterid': str,
        }
    )

    # read which workcenters can support which processes
    workcenter_eligibility = pd.read_csv(
        workcenter_eligibility_path,
        dtype={
            'workcenterid': str,
            'process': str
        }
    )

    # read labor capacity by process
    labor_capacity = pd.read_csv(
        labor_capacity_path,
        dtype={
            'labor_pool': str
        }
    )

    # read which labor pools can support which processes
    labor_eligibility = pd.read_csv(
        labor_eligibility_path,
        dtype={
            'labor_pool': str,
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
    df['work_content_days'] = (
        df['qty'] * df['cycle_time_days']
    )

    # burn-in in sets of 4
    df.loc[df['sequence'] == 3, 'work_content_days'] = np.ceil(
        df.loc[df['sequence'] == 3, 'qty'] / 4
    )

    # fqc set to 1 day for all batch quantities
    df.loc[df['sequence'] == 4, 'work_content_days'] = 1

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

    return (
        df,
        non_production_orders,
        workcenters,
        workcenter_eligibility,
        labor_capacity,
        labor_eligibility
    )
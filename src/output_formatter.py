import pandas as pd
from calendar_utils import (
    add_working_days_floor,
    add_working_days_ceiling
)

def build_non_production_summary(non_production_orders):
    """
    converts non production rows into summary format
    """

    if non_production_orders.empty:
        return pd.DataFrame()

    output = non_production_orders.copy()

    # format dates
    output['due_date'] = pd.to_datetime(output['date']).dt.date

    if 'earlieststartdate' in output.columns:
        output['earlieststartdate'] = pd.to_datetime(
            output['earlieststartdate']
        ).dt.date

    # match summary table structure
    output['batchid'] = ''
    output['planned_batch_start_date'] = output['due_date']
    output['planned_batch_end_date'] = output['due_date']
    output['max_days_late'] = 0
    output['status'] = 'passthrough'

    desired_columns = [
        'batchid',
        'salesid',
        'accountnum',
        'custname',
        'itemid',
        'product',
        'process',
        'sequence',
        'assigned_resource',
        'assigned_workcenter',
        'qty',
        'priority',
        'earlieststartdate',
        'due_date',
        'total_production_days',
        'start_day',
        'end_day',
        'planned_start_date',
        'planned_end_date',
        'days_late',
        'status'
    ]

    existing_columns = [col for col in desired_columns if col in output.columns]

    return output[existing_columns]

def format_operations_schedule(schedule_df, schedule_start_date):
    """
    convert scheduler output start and end dates into actual calendar dates
    start date uses floor logic to mark when operation starts
    end date uses ceiling logic to mark when operation ends
    """

    output = schedule_df.copy()

    # convert solver day offsets to calendar dates
    output['planned_start_date'] = output['start_day'].apply(
        lambda x: add_working_days_floor(schedule_start_date, x)
    )

    output['planned_end_date'] = output['end_day'].apply(
        lambda x: add_working_days_ceiling(schedule_start_date, x)
    )

    # normalize date format
    output['due_date'] = pd.to_datetime(output['due_date']).dt.date
    output['earlieststartdate'] = pd.to_datetime(output['earlieststartdate']).dt.date

    # calculate days_late against due_date
    output['days_late'] = (
        pd.to_datetime(output['planned_end_date']) -
        pd.to_datetime(output['due_date'])
    ).dt.days.clip(lower=0)

    # add column for status based on days_late
    output['status'] = output['days_late'].apply(
        lambda x: 'late' if x > 0 else 'on time'
    )

    # reorder columns for readability
    desired_columns = [
        'batchid',
        'salesid',
        'accountnum',
        'custname',
        'itemid',
        'product',
        'sequence',
        'qty',
        'priority',
        'earlieststartdate',
        'due_date',
        'total_production_days',
        'start_day',
        'end_day',
        'planned_start_date',
        'planned_end_date',
        'days_late',
        'status'
    ]

    existing_columns = [col for col in desired_columns if col in output.columns]
    remaining_columns = [col for col in output.columns if col not in existing_columns]

    output = output[existing_columns + remaining_columns]

    # sort operations in chronological schedule order
    output = output.sort_values(
        [
            'start_day',
            'assigned_resource',
            'assigned_workcenter',
            'batchid',
            'sequence'
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True
        ],
        na_position='last'
    ).reset_index(drop=True)

    return output


def build_order_summary(operations_schedule_df):
    """
    summary of operations schedule by batch showing planned start and end date for full sequence of operations
    uses batchid to keep identical salesid/itemid combinations separate in the list
    """

    summary = (
        operations_schedule_df
        .groupby(
            [
                'batchid',
                'salesid',
                'accountnum',
                'custname',
                'itemid',
                'product'
            ],
            as_index=False
        )
        .agg(
            planned_batch_start_date=('planned_start_date', 'min'),
            planned_batch_end_date=('planned_end_date', 'max'),
            earlieststartdate=('earlieststartdate', 'max'),
            due_date=('due_date', 'max'),
            priority=('priority', 'min'),
            prod=('prod', 'max'),
            qty=('qty', 'max'),
            total_batch_production_days=('total_production_days', 'sum'),
            max_days_late=('days_late', 'max')
        )
    )

    summary['status'] = summary['max_days_late'].apply(
        lambda x: 'late' if x > 0 else 'on time'
    )

    summary = summary.sort_values(
        [
            'planned_batch_start_date',
            'priority',
            'due_date',
            'qty',
            'batchid'
        ],
        ascending=[
            True,
            True,
            True,
            False,
            True
        ]
    ).reset_index(drop=True)

    return summary
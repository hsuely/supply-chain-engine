from pathlib import Path
import time
import pandas as pd

from data_loader import load_and_prepare_data
from scheduler import build_schedule
from output_formatter import (
    format_operations_schedule,
    build_non_production_summary,
    build_order_summary
)


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / 'outputs'

# Change this to the date you want the schedule to begin.
# Must be parseable as a date.
SCHEDULE_START_DATE = '2026-08-17'


def add_working_days(start_date, days_to_add):
    """
    Add working days to a date.
    Weekends are skipped.
    """

    current_date = pd.Timestamp(start_date).normalize()
    days_added = 0

    while days_added < days_to_add:
        current_date += pd.Timedelta(days=1)

        if current_date.weekday() < 5:
            days_added += 1

    return current_date.date()


def summarize_production_down_days(operations_schedule_df, schedule_start_date):
    """
    Print days and date ranges where no production activity is scheduled.

    A workday is considered active if at least one operation overlaps any part
    of that day.
    """

    if operations_schedule_df.empty:
        print('\n=== PRODUCTION DOWN DAYS ===')
        print('No scheduled operations found.')
        return

    first_day = int(operations_schedule_df['start_day'].min())
    last_day = int(operations_schedule_df['end_day'].max())

    down_days = []

    for day in range(first_day, last_day):
        active_operations = operations_schedule_df[
            (operations_schedule_df['start_day'] < day + 1) &
            (operations_schedule_df['end_day'] > day)
        ]

        if active_operations.empty:
            down_days.append(day)

    print('\n=== PRODUCTION DOWN DAYS ===')
    print(f'Total production down days: {len(down_days)}')

    if not down_days:
        print('No production down days found.')
        return

    # group consecutive down days into ranges
    down_ranges = []
    range_start = down_days[0]
    previous_day = down_days[0]

    for day in down_days[1:]:
        if day == previous_day + 1:
            previous_day = day
        else:
            down_ranges.append((range_start, previous_day))
            range_start = day
            previous_day = day

    down_ranges.append((range_start, previous_day))

    print('\nProduction down date ranges:')

    for range_start, range_end in down_ranges:
        start_date = add_working_days(schedule_start_date, range_start)
        end_date = add_working_days(schedule_start_date, range_end)

        range_days = range_end - range_start + 1

        if range_start == range_end:
            print(f'- {start_date} ({range_days} day)')
        else:
            print(f'- {start_date} to {end_date} ({range_days} days)')


def main():
    start_time = time.perf_counter()
    OUTPUT_DIR.mkdir(exist_ok=True)

    print('\n=== SCHEDULER STARTED ===')

    # 1. Load and prepare order/routing/resource data
    (
        df,
        non_production_orders,
        workcenters,
        workcenter_eligibility,
        labor_capacity,
        labor_eligibility
    ) = load_and_prepare_data()

    print('Step 1 complete: data loaded and prepared.')
    print(f'- Production operation rows: {len(df)}')
    print(f'- Non-production rows: {len(non_production_orders)}')
    print(f'- Workcenters loaded: {len(workcenters)}')
    print(f'- Workcenter eligibility rows: {len(workcenter_eligibility)}')
    print(f'- Labor capacity rows: {len(labor_capacity)}')
    print(f'- Labor eligibility rows: {len(labor_eligibility)}')

    # 2. Run OR-Tools scheduler
    schedule_results = build_schedule(
        df,
        workcenters,
        workcenter_eligibility,
        labor_capacity,
        labor_eligibility,
        schedule_start_date=SCHEDULE_START_DATE
    )

    raw_schedule_df = pd.DataFrame(schedule_results)

    # 3. Convert solver offset outputs to calendar dates
    operations_schedule_df = format_operations_schedule(
        raw_schedule_df,
        schedule_start_date=SCHEDULE_START_DATE
    )

    # 4. Build batch-level summary
    scheduled_summary_df = build_order_summary(operations_schedule_df)
    non_production_summary_df = build_non_production_summary(non_production_orders)

    order_summary_df = pd.concat(
        [scheduled_summary_df, non_production_summary_df],
        ignore_index=True,
        sort=False
    )

    order_summary_df = order_summary_df.sort_values(
        [
            'planned_batch_start_date',
            'priority',
            'due_date',
            'prod',
            'salesid',
            'itemid'
        ],
        ascending=[
            True,
            True,
            True,
            False,
            True,
            True
        ]
    ).reset_index(drop=True)

    # reorder order summary columns before printing and exporting
    order_summary_columns = [
        'batchid',
        'salesid',
        'accountnum',
        'custname',
        'itemid',
        'product',
        'qty',
        'priority',
        'prod',
        'earlieststartdate',
        'planned_batch_start_date',
        'planned_batch_end_date',
        'planned_ship_date',
        'due_date',
        'total_work_content_days',
        'total_duration_days',
        'max_days_late',
        'status'
    ]

    existing_columns = [
        col
        for col in order_summary_columns
        if col in order_summary_df.columns
    ]

    remaining_columns = [
        col
        for col in order_summary_df.columns
        if col not in existing_columns
    ]

    order_summary_df = order_summary_df[
        existing_columns + remaining_columns
    ]

    print('\n=== ORDER SUMMARY ===')
    print(
        order_summary_df[
            [
                'batchid',
                'salesid',
                'custname',
                'itemid',
                'product',
                'qty',
                'priority',
                'earlieststartdate',
                'planned_batch_start_date',
                'planned_batch_end_date',
                'planned_ship_date',
                'due_date',
                'total_work_content_days',
                'total_duration_days',
                'max_days_late',
                'status'
            ]
        ]
    )

    # 5. Export files
    raw_schedule_path = OUTPUT_DIR / 'raw_solver_schedule.csv'
    operations_schedule_path = OUTPUT_DIR / 'operations_schedule.csv'
    order_summary_path = OUTPUT_DIR / 'order_summary.csv'

    raw_schedule_df.to_csv(raw_schedule_path, index=False)
    operations_schedule_df.to_csv(operations_schedule_path, index=False)
    order_summary_df.to_csv(order_summary_path, index=False)

    print('Step 8 complete: output files exported.')

    print('\n=== FILES CREATED ===')
    print(f'Raw solver schedule: {raw_schedule_path}')
    print(f'Operations schedule: {operations_schedule_path}')
    print(f'Order summary: {order_summary_path}')

    end_time = time.perf_counter()
    runtime_seconds = end_time - start_time
    print(f'\nTotal runtime: {runtime_seconds:.2f} seconds')

    summarize_production_down_days(
        operations_schedule_df,
        schedule_start_date=SCHEDULE_START_DATE
    )


if __name__ == '__main__':
    main()
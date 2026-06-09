from pathlib import Path
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
SCHEDULE_START_DATE = '2026-06-08'


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Load and prepare order/routing data
    df, non_production_orders, capacity = load_and_prepare_data()

    print('\n=== LOADED DATA ===')
    print(
        df[
            [
                'batchid',
                'salesid',
                'accountnum',
                'custname',
                'itemid',
                'product',
                'qty',
                'priority',
                'earlieststartdate',
                'date',
                'sequence',
                'total_production_days'
            ]
        ]
    )

    # 2. Run OR-Tools scheduler
    schedule_results = build_schedule(
        df,
        schedule_start_date=SCHEDULE_START_DATE
    )

    raw_schedule_df = pd.DataFrame(schedule_results)

    print('\n=== RAW SOLVER SCHEDULE ===')
    print(
        raw_schedule_df[
            [
                'batchid',
                'salesid',
                'itemid',
                'sequence',
                'qty',
                'priority',
                'earlieststartdate',
                'due_date',
                'total_production_days',
                'start_day',
                'end_day'
            ]
        ]
    )

    # 3. Convert solver offset outputs to calendar dates
    operations_schedule_df = format_operations_schedule(
        raw_schedule_df,
        schedule_start_date=SCHEDULE_START_DATE
    )

    print('\n=== OPERATIONS SCHEDULE ===')
    print(
        operations_schedule_df[
            [
                'batchid',
                'salesid',
                'custname',
                'itemid',
                'product',
                'sequence',
                'qty',
                'priority',
                'earlieststartdate',
                'due_date',
                'planned_start_date',
                'planned_end_date',
                'days_late',
                'status',
            ]
        ]
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
                'due_date',
                'planned_batch_start_date',
                'planned_batch_end_date',
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

    print('\n=== FILES CREATED ===')
    print(f'Raw solver schedule: {raw_schedule_path}')
    print(f'Operations schedule: {operations_schedule_path}')
    print(f'Order summary: {order_summary_path}')


if __name__ == '__main__':
    main()
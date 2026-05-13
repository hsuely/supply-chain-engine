from ortools.sat.python import cp_model
import pandas as pd


SCALE = 100

def working_day_offset(schedule_start_date, target_date):
    """
    convert dates into offsets to combined with the scale into solver time units
    removes weekends
    if earlieststartdate is empty, the batch is ready for production
    """

    if pd.isna(target_date):
        return 0

    start = pd.Timestamp(schedule_start_date).normalize()
    target = pd.Timestamp(target_date).normalize()

    if target <= start:
        return 0

    current = start
    offset = 0

    while current < target:
        current += pd.Timedelta(days=1)

        if current.weekday() < 5:
            offset += 1

    return offset


def build_schedule(df, schedule_start_date):

    model = cp_model.CpModel()

    df = df.copy()

    # convert offset into solver time units
    df['duration_int'] = (df['total_production_days'] * SCALE).astype(int)

    # convert earlieststartdate to offset equivalent, or 0 if empty
    df['earliest_start_offset'] = df['earlieststartdate'].apply(
        lambda x: working_day_offset(schedule_start_date, x)
    )

    df['earliest_start_int'] = (
        df['earliest_start_offset'] * SCALE
    ).astype(int)

    # convert due date to offset equivalent
    df['due_date_offset'] = df['date'].apply(
        lambda x: working_day_offset(schedule_start_date, x)
    )

    df['due_date_int'] = (
        df['due_date_offset'] * SCALE
    ).astype(int)

    # set maximum possible schedule length, maximum earlieststartdate + duration of all batches + 10 day buffer
    horizon = int(df['duration_int'].sum() + df['earliest_start_int'].max() + 1000)

    tasks = {}
    machine_to_intervals = {}
    batch_intervals = []
    batch_info = {}
    """
    tasks = each batch operation step and it's parameters (start, end, interval, duration)
    machine_to_intervals = maps processes to intervals, helps with ensuring no or limited overlap to prevent overallocation of resources
    batch_intervals = sets a full interval for a batch from start to end of all tasks
    batch_info = provides the key data for a batch for easy access within the solver
    """

    # create one task for each batch's routing rows
    for idx, row in df.iterrows():
        batchid = row['batchid']
        sequence = int(row['sequence'])
        work_center = f'Process{sequence}'

        start = model.NewIntVar(0, horizon, f'start_{idx}')
        end = model.NewIntVar(0, horizon, f'end_{idx}')

        interval = model.NewIntervalVar(
            start,
            int(row['duration_int']),
            end,
            f'interval_{idx}'
        )

        tasks[idx] = {
            'batchid': batchid,
            'sequence': sequence,
            'start': start,
            'end': end,
            'interval': interval,
            'duration': int(row['duration_int'])
        }

        machine_to_intervals.setdefault(work_center, []).append(interval)

    # rule: each batch must be run in sequence until fully complete
    for batchid in df['batchid'].unique():
        batch_rows = df[df['batchid'] == batchid].sort_values('sequence')

        for i in range(len(batch_rows) - 1):
            current_idx = batch_rows.index[i]
            next_idx = batch_rows.index[i + 1]

            model.Add(
                tasks[next_idx]['start'] == tasks[current_idx]['end']
            )

    # rule: first step of a batch cannot be before its earliest start date
    for batchid in df['batchid'].unique():
        batch_rows = df[df['batchid'] == batchid].sort_values('sequence')

        first_idx = batch_rows.index[0]
        last_idx = batch_rows.index[-1]

        earliest_start_int = int(batch_rows['earliest_start_int'].max())
        due_date_int = int(batch_rows['due_date_int'].min())
        total_duration_int = int(batch_rows['duration_int'].sum())

        model.Add(
            tasks[first_idx]['start'] >= earliest_start_int
        )

        # create batch-level interval spanning all operations
        batch_interval = model.NewIntervalVar(
            tasks[first_idx]['start'],
            total_duration_int,
            tasks[last_idx]['end'],
            f'batch_interval_{batchid}'
        )

        batch_intervals.append(batch_interval)

        batch_info[batchid] = {
            'first_idx': first_idx,
            'last_idx': last_idx,
            'due_date_int': due_date_int,
            'priority': int(batch_rows['priority'].min()),
            'qty': int(batch_rows['qty'].max())
        }

    # rule: no two batches can run at the same time
    model.AddNoOverlap(batch_intervals)

    # rule: no overlap in processes (enforced by earlier rule, but will matter with additional resources)
    for work_center, intervals in machine_to_intervals.items():
        model.AddNoOverlap(intervals)

    # objective_terms = optimization paramters
    # optimze by priority, due dates, and quantity
    # minimize completion score and lateness score
    objective_terms = []

    max_priority = int(df['priority'].max())

    for batchid, info in batch_info.items():
        batch_end = tasks[info['last_idx']]['end']

        # lower priority number = more important.
        priority_strength = max_priority - info['priority'] + 1

        # quantity acts as a tiebreaker
        qty_strength = info['qty']

        # completion weight encourages important/high-qty jobs to finish earlier
        completion_weight = (priority_strength * 1000) + qty_strength

        # completion score
        objective_terms.append(completion_weight * batch_end)

        # lateness variable is batch_end - due_date (unless it's early)
        lateness = model.NewIntVar(0, horizon, f'lateness_{batchid}')

        model.Add(lateness >= batch_end - info['due_date_int'])
        model.Add(lateness >= 0)

        # lateness_weight scaled 100x over completion_weight
        lateness_weight = priority_strength * 100000

        # lateness score
        objective_terms.append(lateness_weight * lateness)

    # set makespan variable
    makespan = model.NewIntVar(0, horizon, 'makespan')

    # gather final steps for each batch
    final_steps = [
        tasks[info['last_idx']]['end']
        for info in batch_info.values()
    ]

    # set makespan to last final_step
    model.AddMaxEquality(makespan, final_steps)

    # add makespan to objective_terms to minimize total schedule length
    objective_terms.append(makespan)
    model.Minimize(sum(objective_terms))

    # solver object
    solver = cp_model.CpSolver()

    # attempts solving the schedule storing best outcome
    status = solver.Solve(model)

    print('Solver status:', solver.StatusName(status))

    # raise error if schedule is not optimal or feasible
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise ValueError('No feasible schedule found')

    # set empty results list
    results = []

    # output results
    for idx, row in df.iterrows():
        results.append({
            'batchid': row['batchid'],
            'salesid': row['salesid'],
            'accountnum': row['accountnum'],
            'custname': row['custname'],
            'itemid': row['itemid'],
            'product': row['product'],
            'sequence': row['sequence'],
            'qty': row['qty'],
            'priority': row['priority'],
            'earlieststartdate': row['earlieststartdate'],
            'due_date': row['date'],
            'total_production_days': row['total_production_days'],
            'start_day': solver.Value(tasks[idx]['start']) / SCALE,
            'end_day': solver.Value(tasks[idx]['end']) / SCALE
        })

    return results
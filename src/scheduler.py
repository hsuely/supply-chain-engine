from ortools.sat.python import cp_model
import pandas as pd


SCALE = 4

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


def build_schedule(df,
                   resources,
                   workcenters,
                   resource_process_eligibility,
                   schedule_start_date):

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
    buffer_days = 10

    horizon = int(
        df['duration_int'].sum()
        + df['earliest_start_int'].max()
        + buffer_days * SCALE
    )

    tasks = {}
    resource_to_intervals = {}
    workcenter_pool_to_intervals = {}
    workcenter_pool_to_demands = {}
    batch_info = {}

    """
    tasks = each batch operation step and its solver parameters
    resource_to_intervals = maps labor resources to assigned operation intervals
    workcenter_to_intervals = maps physical workcenters to assigned operation intervals
    workcenter_to_demands = demand values used for workcenter cumulative capacity
    batch_info = key batch-level data used in the objective function
    """

    # calculate pooled workcenter capacity by process
    workcenter_pool_capacity = (
        workcenters
        .groupby('process')['capacity']
        .sum()
        .astype(int)
        .to_dict()
    )

    # create one task for each batch's routing rows
    for idx, row in df.iterrows():
        batchid = row['batchid']
        sequence = int(row['sequence'])
        process = row['process']

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
            'process': process,
            'sequence': sequence,
            'start': start,
            'end': end,
            'interval': interval,
            'duration': int(row['duration_int']),
            'resource_choices': {},
            'assigned_workcenter_pool': None
        }

        # determine list of eligible resources for this task
        eligible_resources = (
            resource_process_eligibility[
                resource_process_eligibility['process'] == process
                ]['resourceid']
            .tolist()
        )

        # selects an eligible resource for the task
        if int(row['labor_required']) > 0:
            if not eligible_resources:
                raise ValueError(f'No eligible resources found for process: {process}')

            resource_presence_vars = []

            for resourceid in eligible_resources:
                resource_presence = model.NewBoolVar(
                    f'assigned_{idx}_{resourceid}'
                )

                resource_interval = model.NewOptionalIntervalVar(
                    start,
                    int(row['duration_int']),
                    end,
                    resource_presence,
                    f'interval_{idx}_{resourceid}'
                )

                tasks[idx]['resource_choices'][resourceid] = resource_presence

                resource_to_intervals.setdefault(resourceid, []).append(
                    resource_interval
                )

                resource_presence_vars.append(resource_presence)

            model.AddExactlyOne(resource_presence_vars)

        # assigns workcenter for the task if the process requires it
        workcenter_capacity = int(workcenter_pool_capacity.get(process, 0))

        if workcenter_capacity > 0:
            workcenter_pool_to_intervals.setdefault(process, []).append(interval)
            workcenter_pool_to_demands.setdefault(process, []).append(1)
            tasks[idx]['assigned_workcenter_pool'] = f'{process}_pool'

    # rule: each batch must be run in sequence until fully complete
    for batchid in df['batchid'].unique():
        batch_rows = df[df['batchid'] == batchid].sort_values('sequence')

        for i in range(len(batch_rows) - 1):
            current_idx = batch_rows.index[i]
            next_idx = batch_rows.index[i + 1]

            model.Add(
                tasks[next_idx]['start'] >= tasks[current_idx]['end']
            )

    # rule: first step of a batch cannot be before its earliest start date
    for batchid in df['batchid'].unique():
        batch_rows = df[df['batchid'] == batchid].sort_values('sequence')

        first_idx = batch_rows.index[0]
        last_idx = batch_rows.index[-1]

        earliest_start_int = int(batch_rows['earliest_start_int'].max())
        due_date_int = int(batch_rows['due_date_int'].min())

        model.Add(
            tasks[first_idx]['start'] >= earliest_start_int
        )

        batch_info[batchid] = {
            'first_idx': first_idx,
            'last_idx': last_idx,
            'due_date_int': due_date_int,
            'priority': int(batch_rows['priority'].min()),
            'qty': int(batch_rows['qty'].max())
        }

    # rule: each labor resource can only work on one operation at a time
    for resourceid, intervals in resource_to_intervals.items():
        model.AddNoOverlap(intervals)

    # rule: each workcenter process pool cannot exceed its total capacity
    for process, intervals in workcenter_pool_to_intervals.items():
        capacity_value = int(workcenter_pool_capacity[process])

        if capacity_value == 1:
            model.AddNoOverlap(intervals)
        else:
            demands = workcenter_pool_to_demands[process]

            model.AddCumulative(
                intervals,
                demands,
                capacity_value
            )

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

    # help with runtime, start early orders first
    model.AddDecisionStrategy(
        [tasks[idx]['start'] for idx in tasks],
        cp_model.CHOOSE_LOWEST_MIN,
        cp_model.SELECT_MIN_VALUE
    )

    # solver object
    solver = cp_model.CpSolver()

    # solver time limits
    solver.parameters.max_time_in_seconds = 30
    solver.parameters.num_search_workers = 8

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
        assigned_resource = None

        for resourceid, presence_var in tasks[idx]['resource_choices'].items():
            if solver.Value(presence_var) == 1:
                assigned_resource = resourceid
                break

        assigned_workcenter = None

        assigned_workcenter = tasks[idx]['assigned_workcenter_pool']

        results.append({
            'batchid': row['batchid'],
            'salesid': row['salesid'],
            'accountnum': row['accountnum'],
            'custname': row['custname'],
            'itemid': row['itemid'],
            'product': row['product'],
            'sequence': row['sequence'],
            'process': row['process'],
            'assigned_resource': assigned_resource,
            'assigned_workcenter': assigned_workcenter,
            'qty': row['qty'],
            'priority': row['priority'],
            'prod': row['prod'],
            'earlieststartdate': row['earlieststartdate'],
            'due_date': row['date'],
            'total_production_days': row['total_production_days'],
            'start_day': solver.Value(tasks[idx]['start']) / SCALE,
            'end_day': solver.Value(tasks[idx]['end']) / SCALE
        })

    return results
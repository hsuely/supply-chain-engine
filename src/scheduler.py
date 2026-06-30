from ortools.sat.python import cp_model
import pandas as pd
import itertools
import math


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
                   workcenters,
                   workcenter_eligibility,
                   labor_capacity,
                   labor_eligibility,
                   schedule_start_date):

    model = cp_model.CpModel()

    df = df.copy()

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

    # use a conservative horizon based on the slowest possible interpretation
    max_work_content_int = (
            df['work_content_days'].fillna(0).sum() * SCALE * 2
    )

    horizon = int(
        max_work_content_int
        + df['earliest_start_int'].max()
        + buffer_days * SCALE
    )

    tasks = {}
    labor_pool_to_intervals = {}
    labor_pool_to_demands = {}
    workcenter_to_intervals = {}
    workcenter_to_demands = {}
    batch_info = {}

    """
    tasks = each batch operation step and its solver parameters
    labor_pool_to_intervals = maps labor process pools to operation intervals
    labor_pool_to_demands = demand values used for labor pool capacity
    workcenter_to_intervals = maps physical workcenters to operation intervals
    workcenter_to_demands = demand values used for physical workcenter capacity
    batch_info = key batch-level data used in the objective function
    """

    # calculate physical workcenter capacity by workcenterid
    workcenter_capacity = (
        workcenters
        .set_index('workcenterid')['capacity']
        .astype(int)
        .to_dict()
    )

    def get_eligible_workcenters(process):
        """
        Get physical workcenters that can support a process.
        """

        eligible_workcenters = (
            workcenter_eligibility[
                workcenter_eligibility['process'] == process
                ]['workcenterid']
            .dropna()
            .unique()
            .tolist()
        )

        return eligible_workcenters

    # calculate pooled labor capacity by process
    capacity_scale = 4

    labor_pool_capacity = (
        labor_capacity
        .set_index('labor_pool')['daily_capacity']
        .mul(capacity_scale)
        .round()
        .astype(int)
        .to_dict()
    )

    def generate_labor_modes(process):
        """
        Generate valid labor pool combinations for a process.

        Each mode contains:
        - mode_name
        - labor_pools
        - labor_rate
        """

        eligible_pools = (
            labor_eligibility[
                labor_eligibility['process'] == process
                ]['labor_pool']
            .dropna()
            .unique()
            .tolist()
        )

        if not eligible_pools:
            return []

        modes = []

        for combination_size in range(1, len(eligible_pools) + 1):
            for labor_pool_combo in itertools.combinations(
                    eligible_pools,
                    combination_size
            ):
                labor_rate = sum(
                    labor_pool_capacity[labor_pool]
                    for labor_pool in labor_pool_combo
                )

                mode_name = '+'.join(labor_pool_combo)

                modes.append({
                    'mode_name': mode_name,
                    'labor_pools': list(labor_pool_combo),
                    'labor_rate': labor_rate
                })

        return modes

    # create one task for each batch's routing rows
    for idx, row in df.iterrows():
        batchid = row['batchid']
        sequence = int(row['sequence'])
        process = row['process']
        work_content_days = float(row['work_content_days'])

        start = model.NewIntVar(0, horizon, f'start_{idx}')
        end = model.NewIntVar(0, horizon, f'end_{idx}')

        tasks[idx] = {
            'batchid': batchid,
            'process': process,
            'sequence': sequence,
            'start': start,
            'end': end,
            'interval': None,
            'duration': None,
            'mode_choices': {},
            'workcenter_choices': {},
            'assigned_labor_pool': None,
            'assigned_workcenter_pool': None
        }

        # labor-driven operation: generate optional labor modes
        if int(row['labor_required']) > 0:
            labor_modes = generate_labor_modes(process)

            mode_presence_vars = []

            for mode in labor_modes:
                mode_name = mode['mode_name']
                labor_rate = mode['labor_rate']

                duration_int = int(
                    math.ceil(
                        (work_content_days * SCALE * capacity_scale)
                        / labor_rate
                    )
                )

                presence = model.NewBoolVar(
                    f'presence_{idx}_{mode_name}'
                )

                optional_interval = model.NewOptionalIntervalVar(
                    start,
                    duration_int,
                    end,
                    presence,
                    f'interval_{idx}_{mode_name}'
                )

                tasks[idx]['mode_choices'][mode_name] = {
                    'presence': presence,
                    'duration': duration_int,
                    'labor_pools': mode['labor_pools']
                }

                mode_presence_vars.append(presence)

                for labor_pool in mode['labor_pools']:
                    labor_pool_to_intervals.setdefault(
                        labor_pool,
                        []
                    ).append(optional_interval)

                    labor_pool_to_demands.setdefault(
                        labor_pool,
                        []
                    ).append(labor_pool_capacity[labor_pool])

            model.AddExactlyOne(mode_presence_vars)

        # non-labor operation: fixed duration interval
        else:
            duration_int = int(
                math.ceil(work_content_days * SCALE)
            )

            interval = model.NewIntervalVar(
                start,
                duration_int,
                end,
                f'interval_{idx}'
            )

            tasks[idx]['interval'] = interval
            tasks[idx]['duration'] = duration_int

            # if non-labor task requires workcenter, choose one eligible physical workcenter
            if int(row['workcenter_required']) > 0:
                eligible_workcenters = get_eligible_workcenters(process)

                if not eligible_workcenters:
                    raise ValueError(
                        f'No eligible workcenters found for process: {process}'
                    )

                workcenter_presence_vars = []

                for workcenterid in eligible_workcenters:
                    if workcenterid not in workcenter_capacity:
                        raise ValueError(
                            f'No capacity found for workcenter: {workcenterid}'
                        )

                    workcenter_presence = model.NewBoolVar(
                        f'workcenter_{idx}_{workcenterid}'
                    )

                    optional_workcenter_interval = model.NewOptionalIntervalVar(
                        start,
                        duration_int,
                        end,
                        workcenter_presence,
                        f'workcenter_interval_{idx}_{workcenterid}'
                    )

                    tasks[idx]['workcenter_choices'][workcenterid] = {
                        'presence': workcenter_presence,
                        'workcenterid': workcenterid,
                        'mode_name': None
                    }

                    workcenter_presence_vars.append(workcenter_presence)

                    workcenter_to_intervals.setdefault(
                        workcenterid,
                        []
                    ).append(optional_workcenter_interval)

                    workcenter_to_demands.setdefault(
                        workcenterid,
                        []
                    ).append(1)

                model.AddExactlyOne(workcenter_presence_vars)

        # labor-driven tasks may also require one eligible physical workcenter
        if int(row['labor_required']) > 0 and int(row['workcenter_required']) > 0:
            eligible_workcenters = get_eligible_workcenters(process)

            if not eligible_workcenters:
                raise ValueError(
                    f'No eligible workcenters found for process: {process}'
                )

            for mode_name, mode_info in tasks[idx]['mode_choices'].items():
                mode_presence = mode_info['presence']
                duration_int = mode_info['duration']

                workcenter_presence_vars = []

                for workcenterid in eligible_workcenters:
                    if workcenterid not in workcenter_capacity:
                        raise ValueError(
                            f'No capacity found for workcenter: {workcenterid}'
                        )

                    workcenter_presence = model.NewBoolVar(
                        f'workcenter_{idx}_{mode_name}_{workcenterid}'
                    )

                    optional_workcenter_interval = model.NewOptionalIntervalVar(
                        start,
                        duration_int,
                        end,
                        workcenter_presence,
                        f'workcenter_interval_{idx}_{mode_name}_{workcenterid}'
                    )

                    tasks[idx]['workcenter_choices'][
                        f'{mode_name}|{workcenterid}'
                    ] = {
                        'presence': workcenter_presence,
                        'workcenterid': workcenterid,
                        'mode_name': mode_name
                    }

                    workcenter_presence_vars.append(workcenter_presence)

                    workcenter_to_intervals.setdefault(
                        workcenterid,
                        []
                    ).append(optional_workcenter_interval)

                    workcenter_to_demands.setdefault(
                        workcenterid,
                        []
                    ).append(1)

                model.Add(
                    sum(workcenter_presence_vars) == mode_presence
                )

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

    # rule: each labor process pool cannot exceed its available capacity
    for labor_pool, intervals in labor_pool_to_intervals.items():
        capacity_value = int(labor_pool_capacity[labor_pool])
        demands = labor_pool_to_demands[labor_pool]

        model.AddCumulative(
            intervals,
            demands,
            capacity_value
        )

    # rule: each physical workcenter cannot exceed its available capacity
    for workcenterid, intervals in workcenter_to_intervals.items():
        capacity_value = int(workcenter_capacity[workcenterid])

        if capacity_value == 1:
            model.AddNoOverlap(intervals)
        else:
            demands = workcenter_to_demands[workcenterid]

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
        assigned_resource = tasks[idx]['assigned_labor_pool']

        if tasks[idx]['mode_choices']:
            for mode_name, mode_info in tasks[idx]['mode_choices'].items():
                if solver.Value(mode_info['presence']) == 1:
                    assigned_resource = mode_name
                    break

        assigned_workcenter = tasks[idx]['assigned_workcenter_pool']

        if tasks[idx]['workcenter_choices']:
            for choice_name, choice_info in tasks[idx]['workcenter_choices'].items():
                if solver.Value(choice_info['presence']) == 1:
                    assigned_workcenter = choice_info['workcenterid']
                    break

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
            'work_content_days': row['work_content_days'],
            'duration_days': (solver.Value(tasks[idx]['end']) - solver.Value(tasks[idx]['start'])) / SCALE,
            'start_day': solver.Value(tasks[idx]['start']) / SCALE,
            'end_day': solver.Value(tasks[idx]['end']) / SCALE
        })

    return results
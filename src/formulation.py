import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from src.utils import pace_to_str, str_to_pace, format_seconds

def optimal_schedule(route, skills, 
                     time_limit=60, 
                     toll=1e-6, 
                     save=True, 
                     max_drive_change=5, 
                     min_start_rest=20,
                     max_start_rest=33,
                     max_long_rest=7,
                     quadratic=False):
    
    segments = list(route.index)
    people = list(skills.index)
    n_segments = len(segments)
    n_people = len(people)

    m = gp.Model("RelayRunnerAssignment")
    m.setParam('MIPGap', toll)
    m.Params.TimeLimit = time_limit

    # --- Variables ---
    x = m.addVars(segments, people, vtype=GRB.BINARY, name="run")
    d = m.addVars(segments, people, vtype=GRB.BINARY, name="drive")
    z = m.addVars(segments, people, vtype=GRB.BINARY, name="same_driver")
    r = {}
    for j in range(n_people):
        if skills.loc[j, 'min_long_rest'] == 0:
            continue
        for s in range(min_start_rest, max_start_rest):
            r[s, j] = m.addVar(vtype=GRB.BINARY, name=f"rest_block_{s}_{j}")

    # Objective: minimize total time
    if quadratic:
        total_distance = m.addVars(people, lb=0.0, name="total_distance")
        total_distance_night = m.addVars(people, lb=0.0, name="total_distance_night")
        total_D_plus = m.addVars(people, lb=0.0, name="total_Dplus")
        total_D_minus = m.addVars(people, lb=0.0, name="total_Dminus")
        time_per_runner = m.addVars(people, lb=0.0, name="total_time")

        for j in people:
            m.addConstr(total_distance[j] == gp.quicksum(x[i, j] * (route.loc[i, "Distance"] / (1 + route.loc[i, "Bike"])) for i in segments))
            m.addConstr(total_distance_night[j] == gp.quicksum(x[i, j] * (route.loc[i, "Distance"] / (1 + route.loc[i, "Bike"])) for i in segments if route.loc[i, "night"] == 1))
            m.addConstr(total_D_plus[j] == gp.quicksum(x[i, j] * (route.loc[i, "D+"] / (1 + route.loc[i, "Bike"])) for i in segments))
            m.addConstr(total_D_minus[j] == gp.quicksum(x[i, j] * (route.loc[i, "D-"] / (1 + route.loc[i, "Bike"])) for i in segments))

        for j in people:
            pace_HM = skills.loc[j, "pace_HM"]
            endurance = skills.loc[j, "endurance"]
            p_night = skills.loc[j, "p_night"]
            k_plus = skills.loc[j, "K+"]
            k_minus = skills.loc[j, "K-"]
            td = total_distance[j]
            tdn = total_distance_night[j]
            tdp = total_D_plus[j]
            tdm = total_D_minus[j]
            adj_td = td + tdp/100
            base_pace = (pace_HM * (1+ endurance * (adj_td - 21)))
            m.addQConstr(
                time_per_runner[j] == 
                    base_pace * td +
                    p_night * tdn +
                    tdp * k_plus +
                    tdm * k_minus
            )
        m.setObjective(gp.quicksum(time_per_runner[j] for j in people), GRB.MINIMIZE)
    else:
        m.setObjective(
            gp.quicksum(
                x[i, j] * (route.loc[i, "Distance"] / (1 + route.loc[i, "Bike"])) * skills.loc[j, "pace_HM"]
                for i in segments for j in people
            ),
            GRB.MINIMIZE
        )

    # --- Constraints ---

    # CN1: One runner+biker per segment
    for i in segments:
        m.addConstr(gp.quicksum(x[i, j] for j in people) == 1 + route.loc[i, "Bike"])

    # CN2: One driver per segment and not overlapping with runner/biker
    for i in segments:
        m.addConstr(gp.quicksum(d[i, j] for j in people) == 1)
        for j in people:
            m.addConstr(x[i, j] + d[i, j] <= 1)

    # Run1: Running per person limits
    for j in people:
        dist_expr = gp.quicksum(x[i, j] * (route.loc[i, "Distance"] / (1 + route.loc[i, "Bike"])) for i in segments)
        dplus_expr = gp.quicksum(x[i, j] * (route.loc[i, "D+"] / (1 + route.loc[i, "Bike"])) for i in segments)
        m.addConstr(dist_expr >= skills.loc[j, "min_run"])
        m.addConstr(dist_expr <= skills.loc[j, "max_run"])
        m.addConstr(dplus_expr <= skills.loc[j, "max_D+"])

    # Run2: No solo running if requested
    for j in people:
        for i in segments:
            if skills.loc[j, "solo_running"] == 0:
                m.addConstr(x[i, j] <= route.loc[i, "Bike"])

    # Recovery1: Recovery among legs
    for j in people:
        for i in range(n_segments - 2):
            m.addConstr(x[i, j] + x[i + 1, j] <= 1)
            m.addConstr(x[i, j] + x[i + 2, j] <= 1)
        m.addConstr(x[n_segments - 2, j] + x[n_segments - 1, j] <= 1)

    # Recovery2: Sleep shift
    for j in range(n_people):
        if skills.loc[j, 'min_long_rest'] == 0:
            continue
        min_rest = int(skills.loc[j, 'min_long_rest'])
        for s in range(min_start_rest, max_start_rest):
            for i in range(s, s + min_rest):
                if i < n_segments:
                    m.addConstr(d[i, j] + x[i, j] <= 1 - r[s, j])
        m.addConstr(gp.quicksum(r[s, j] for s in range(min_start_rest, max_start_rest)) >= 1)
    
    # Recovery3: Avoid very long rest (always run or drive each max_long_rest segments)
    for j in people:
        for i in range(n_segments - 6):
            m.addConstr(gp.quicksum(x[i + k, j]+d[i + k, j] for k in range(max_long_rest)) >= 1)

    # Drive1: Rest before driving
    for i in range(n_segments - 1):
        for j in people:
            m.addConstr(x[i, j] + d[i + 1, j] <= 1)

    # Drive2: Driving per person limits
    for j in people:
        m.addConstr(gp.quicksum(d[i, j] * route.loc[i, "Distance"] for i in segments) <= skills.loc[j, "max_drive"])

    # Drive3: Night driving
    for i in segments:
        for j in people:
            m.addConstr(d[i, j] <= 1 - route.loc[i, 'night'] + skills.loc[j, 'drive_night'])

    # Drive4: Limit number of driving changes
    for j in people:
        m.addConstr(z[0, j] == d[0, j])
    for i in range(1, n_segments):
        for j in people:
            m.addConstr(z[i, j] >= d[i, j] - d[i - 1, j])
            m.addConstr(z[i, j] <= d[i, j])
            m.addConstr(z[i, j] <= 1 - d[i - 1, j])
    m.addConstr(gp.quicksum(z[i, j] for i in range(1, n_segments) for j in people) <= max_drive_change)

    # --- Solve ---
    m.optimize()

    if m.status == GRB.Status.TIME_LIMIT:
        print("Time limit reached. Using best feasible solution found.")
    elif m.status == GRB.Status.OPTIMAL:
        print("Optimal solution found.")
    elif m.status == GRB.Status.SUBOPTIMAL:
        print("Suboptimal feasible solution found.")
    elif m.status == GRB.Status.INFEASIBLE:
        print("Model is infeasible.")
        return None, None

    # --- Extract results ---
    expected_time = m.ObjVal
    expected_time = int(expected_time // 3600), int((expected_time % 3600) // 60), int(expected_time % 60)
    print(f"Expected Time: {expected_time[0]}h{expected_time[1]:02d}m{expected_time[2]:02d}s")

    assignment, summary = extract_results(route, skills, x, d, r, z, min_start_rest, max_start_rest, save)
    return assignment, summary

def extract_results(route, skills, x, d, r, z, min_start_rest, max_start_rest, save=True):
    segments = list(route.index)
    people = list(skills.index)
    n_segments = len(segments)
    n_people = len(people)

    summary = {
        "Runner": [], "Legs": [], "Distance": [], "D+": [], "D-": [], "Adj. Distance":[], "Driving": [], "Driving shifts": [], "Base pace": []
    }
    for j in people:
        assigned_legs = [i for i in segments if x[i, j].X > 0.5]
        assigned_driving = [i for i in segments if d[i, j].X > 0.5]
        summary["Runner"].append(skills.loc[j, "Runner"])
        summary["Legs"].append(len(assigned_legs))
        summary["Distance"].append(sum(route.loc[i, "Distance"] / (1 + route.loc[i, "Bike"]) for i in assigned_legs))
        summary["D+"].append(int(sum(route.loc[i, "D+"] / (1 + route.loc[i, "Bike"]) for i in assigned_legs)))
        summary["D-"].append(int(sum(route.loc[i, "D-"] / (1 + route.loc[i, "Bike"]) for i in assigned_legs)))
        summary["Adj. Distance"].append(summary["Distance"][-1] + summary["D+"][-1] / 100)
        summary["Driving"].append(sum(route.loc[i, "Distance"] for i in assigned_driving))
        summary["Driving shifts"].append(int(sum(z[i, j].X for i in range(1, n_segments) if z[i, j].X > 0.5)))
        summary["Base pace"].append(pace_to_str(skills.loc[j, "pace_HM"] * (1+skills.loc[j, "endurance"] * (summary["Adj. Distance"][-1] - 21))))


    summary_df = pd.DataFrame(summary)
    summary_df.set_index('Runner', inplace=True)

    if save:
        summary_df.to_csv('results/summary.csv')

    ###############

    assignment = {
        "Segment": [], "Place": [], "From": [], "To": [], "Distance": [], "D+": [], "D-": [],
        "Runner A": [], "Speed A": [], "Runner B": [], "Speed B": [], "Driver": [], "Sleep": [], "D+/km": [], "D-/km": [], "Time": []
    }

    rest_start = {}
    for j in range(n_people):
        if skills.loc[j, 'min_long_rest'] == 0:
            continue
        for s in range(min_start_rest, max_start_rest):
            if (s, j) in r and r[s, j].X > 0.5:
                rest_start[j] = s
                break
    
    t = 6*60*60 # 6 AM start time
    for i in segments:
        assigned_runners = [skills.loc[j, "Runner"] for j in people if x[i, j].X > 0.5]
        assigned_driver = [skills.loc[j, "Runner"] for j in people if d[i, j].X > 0.5][0]
        assignment["Segment"].append(int(route.loc[i, "Segment"]))
        assignment["From"].append(route.loc[i, "From"])
        assignment["To"].append(route.loc[i, "To"])
        assignment["Distance"].append(route.loc[i, "Distance"])
        assignment["D+"].append(route.loc[i, "D+"])
        assignment["D-"].append(route.loc[i, "D-"])
        assignment["D+/km"].append(int(round(route.loc[i, "D+"] / route.loc[i, "Distance"], 0)))
        assignment["D-/km"].append(int(round(route.loc[i, "D-"] / route.loc[i, "Distance"], 0)))
        assignment["Runner A"].append(assigned_runners[0])
        j = skills[skills["Runner"] == assigned_runners[0]].index[0]
        assignment["Speed A"].append(pace_to_str(str_to_pace(summary_df.loc[assigned_runners[0], "Base pace"])+skills.loc[j, "K+"]*assignment["D+/km"][-1]+skills.loc[j, "K-"]*assignment["D-/km"][-1]+skills.loc[j, "p_night"]))
        if len(assigned_runners) > 1:
            assignment["Runner B"].append(assigned_runners[1])
            j = skills[skills["Runner"] == assigned_runners[1]].index[0]
            assignment["Speed B"].append(pace_to_str(str_to_pace(summary_df.loc[assigned_runners[1], "Base pace"])+skills.loc[j, "K+"]*assignment["D+/km"][-1]+skills.loc[j, "K-"]*assignment["D-/km"][-1]+skills.loc[j, "p_night"]))
        else:
            assignment["Runner B"].append("")
            assignment["Speed B"].append("")
        assignment["Driver"].append(assigned_driver)
        assignment["Place"].append(route.loc[i, "Places"])
        sleep = []
        for j in range(n_people):
            if skills.loc[j, 'min_long_rest'] == 0:
                continue
            if i >= rest_start.get(j, -1) and i < rest_start.get(j, -1) + int(skills.loc[j, 'min_long_rest']):
                sleep.append(skills.loc[j, "Runner"])
        assignment["Sleep"].append(sleep)

        distance = route.loc[i, "Distance"]
        if route.loc[i, "Bike"] == 1:
            speed = (str_to_pace(assignment["Speed A"][-1])+str_to_pace(assignment["Speed B"][-1]))/2
        else:
            speed = str_to_pace(assignment["Speed A"][-1])
        t += distance * speed 
        assignment["Time"].append(format_seconds(t))

    assignment_df = pd.DataFrame(assignment)

    rest_block_index = {}
    for j in range(n_people):
        if skills.loc[j, 'min_long_rest'] == 0:
            continue
        for s in range(min_start_rest, n_segments):
            if (s, j) in r and r[s, j].X > 0.5:
                rest_block_index[j] = s
                break

    sleep_by_segment = {i: [] for i in segments}
    for j in range(n_people):
        if skills.loc[j, 'min_long_rest'] == 0 or j not in rest_block_index:
            continue
        s_rest = rest_block_index[j]
        s = s_rest - 1
        while s >= 0 and x[s, j].X < 0.5 and d[s, j].X < 0.5:
            sleep_by_segment[segments[s]].append(skills["Runner"][j])
            s -= 1
        if x[s_rest, j].X < 0.5 and d[s_rest, j].X < 0.5:
            sleep_by_segment[segments[s_rest]].append(skills["Runner"][j])
        s = s_rest + 1
        while s < n_segments and x[s, j].X < 0.5 and d[s, j].X < 0.5:
            sleep_by_segment[segments[s]].append(skills["Runner"][j])
            s += 1
    assignment_df["Sleep"] = assignment_df.index.map(lambda i: sleep_by_segment.get(i, []))
    assignment_df.set_index('Segment', inplace=True)
    if save:
        assignment_df.to_csv('results/schedule.csv')

    return assignment_df, summary_df


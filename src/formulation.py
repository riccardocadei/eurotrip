import pulp
import pandas as pd

def optimal_schedule(route, skills, solver="cbc", time_limit=60, save=True, max_drive_change = 5):
    # --- Sets ---
    segments = list(route.index)
    people = list(skills.index)
    n_segments = len(segments)
    n_people = len(people)
    min_start_rest = 20
    max_start_rest = 33
    max_drive_change = 5

    # --- Problem ---
    prob = pulp.LpProblem("RelayRunnerAssignment", pulp.LpMinimize)

    # --- Variables ---
    x = pulp.LpVariable.dicts("run", [(i, j) for i in segments for j in people], cat="Binary")
    d = pulp.LpVariable.dicts("drive", [(i, j) for i in segments for j in people], cat="Binary")
    z = pulp.LpVariable.dicts("same_driver", [(i, j) for i in segments for j in people], cat="Binary")
    r = {}
    for j in range(n_people):
        if skills.loc[j, 'min_long_rest'] == 0:
            continue
        for s in range(min_start_rest,max_start_rest):
            r[s, j] = pulp.LpVariable(f"rest_block_{s}_{j}", cat='Binary')

    # # --- Auxiliary variables ---
    # total_distance = pulp.LpVariable.dicts("total_distance", people, lowBound=0)
    # total_D_plus = pulp.LpVariable.dicts("total_Dplus", people, lowBound=0)
    # total_D_minus = pulp.LpVariable.dicts("total_Dminus", people, lowBound=0)
    # time_per_runner = pulp.LpVariable.dicts("total_time", people, lowBound=0)
    # for j in people:
    #     prob += total_distance[j] == pulp.lpSum(x[i, j] * (route.loc[i, "Distance"] / (1 + route.loc[i, "Bike"])) for i in segments)
    #     prob += total_D_plus[j] == pulp.lpSum(x[i, j] * (route.loc[i, "D+"] / (1 + route.loc[i, "Bike"])) for i in segments)
    #     prob += total_D_minus[j] == pulp.lpSum(x[i, j] * (route.loc[i, "D-"] / (1 + route.loc[i, "Bike"])) for i in segments)
    # for j in people:
    #     prob += time_per_runner[j] == (
    #         total_distance[j] * ((skills.loc[j, "pace_HM"] + skills.loc[j, "endurance"] * (total_distance[j] - 21)) + skills.loc[j, "p_night"]) +
    #         total_D_plus[j] * skills.loc[j, "k+"] +
    #         total_D_minus[j] * skills.loc[j, "k-"]
    # )
    
    # prob += pulp.lpSum(time_per_runner[j] for j in people) 

    # --- Objective: Minimize total time ---
    prob += pulp.lpSum(
        x[i, j] * (route.loc[i, "Distance"] / (1+route.loc[i, "Bike"])) * (skills.loc[j, "pace_HM"])
        for i in segments for j in people
    )

    # --- Constraints ---

    # 0a. Exactly 1 runner + 1 bike per segment
    for i in segments:
        prob += pulp.lpSum(x[i, j] for j in people) == 1 + route.loc[i, "Bike"]

    # 0b. One driver per segment, and different from runner+biker
    for i in segments:
        prob += pulp.lpSum(d[i, j] for j in people) == 1
        for j in people:
            prob += x[i, j] + d[i, j] <= 1

    # R1. Runner limits (min/max distance and max D+)
    for j in people:
        prob += pulp.lpSum(x[i, j] * (route.loc[i, "Distance"] / (1+route.loc[i, "Bike"])) for i in segments) >= skills.loc[j, "min_run"]
        prob += pulp.lpSum(x[i, j] * (route.loc[i, "Distance"] / (1+route.loc[i, "Bike"])) for i in segments) <= skills.loc[j, "max_run"]
        prob += pulp.lpSum(x[i, j] * (route.loc[i, "D+"] / (1+route.loc[i, "Bike"])) for i in segments) <= skills.loc[j, "max_D+"]

    # S1. Recovery constraint: at least 2 segments between runs
    for j in people:
        for i in range(n_segments - 2):
            prob += x[i, j] + x[i + 1, j] <= 1
            prob += x[i, j] + x[i + 2, j] <= 1
        prob += x[n_segments - 2, j] + x[n_segments - 1, j] <= 1

    # S2. Rest block constraint
    for j in range(n_people):
        if skills.loc[j, 'min_long_rest'] == 0:
            continue
        min_rest = int(skills.loc[j, 'min_long_rest'])
        for s in range(min_start_rest,max_start_rest):
            for i in range(s, s + min_rest):
                prob += d[i, j] + x[i, j] <= 1 - r[s, j]
    for j in range(n_people):
        if skills.loc[j, 'min_long_rest'] == 0:
            continue
        prob += pulp.lpSum(r[s, j] for s in range(min_start_rest,max_start_rest)) >= 1

    # D1. Force to rest before driving
    for i in range(n_segments - 1):
        for j in people:
            prob += x[i, j] + d[i+1, j] <= 1

    # D2. Max driving per person
    for j in people:
        prob += pulp.lpSum(d[i, j] * route.loc[i, "Distance"] for i in segments) <= skills.loc[j, "max_drive"]

    # D3. Night driving constraint 
    for i in segments:
        for j in people:
            prob += d[i, j] <= 1 - route.loc[i, 'night'] + skills.loc[j, 'drive_night']

    # D4. Max driver changes
    for j in people:
        prob += z[0, j] == d[0, j]
    for i in range(1, n_segments):
        for j in people:
            prob += z[i, j] >= d[i, j] - d[i-1, j]
            prob += z[i, j] <= d[i, j]
            prob += z[i, j] <= 1 - d[i-1, j]
    prob += pulp.lpSum(z[i, j] for i in range(1, n_segments) for j in people) <= max_drive_change

    # --- Solve ---
    if solver == "gurobi":
        prob.solve(pulp.GUROBI_CMD(timeLimit=time_limit, msg=True))
    elif solver == "cbc":
        prob.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=True))
    else:
        raise ValueError("Solver not recognized. Use 'gurobi' or 'cbc'.")
    
    # if solution is infeasible, return None
    if prob.status != 1:
        print("Infeasible formulation")
        return None, None
    
    # --- Results ---
    # schedule
    assignment = {
        "Segment": [],
        "Place": [],
        "From": [],
        "To": [],
        "Distance": [],
        "D+": [],
        "D-": [],
        "Runner A": [],
        "Runner B": [],
        "Driver": [],
        "Sleep": [],
    }

    rest_start = {} 
    for j in range(n_people):
        if skills.loc[j, 'min_long_rest'] == 0:
            continue
        for s in range(min_start_rest, max_start_rest):
            if r[s, j].varValue == 1:
                rest_start[j] = s
                break

    for i in segments:
        assigned_runners = [skills.loc[j, "Runner"] for j in people if pulp.value(x[i, j]) == 1]
        assigned_driver = [skills.loc[j, "Runner"] for j in people if pulp.value(d[i, j]) == 1][0]
        assignment["Segment"].append(int(route.loc[i, "Segment"]))
        assignment["From"].append(route.loc[i, "From"])
        assignment["To"].append(route.loc[i, "To"])
        assignment["Distance"].append(route.loc[i, "Distance"])
        assignment["D+"].append(route.loc[i, "D+"])
        assignment["D-"].append(route.loc[i, "D-"])
        assignment["Runner A"].append(assigned_runners[0])
        if len(assigned_runners) > 1:
            assignment["Runner B"].append(assigned_runners[1])
        else:
            assignment["Runner B"].append("")
        assignment["Driver"].append(assigned_driver)
        assignment["Place"].append(route.loc[i, "Places"]) 
        sleep = []
        for j in range(n_people):
            if skills.loc[j, 'min_long_rest'] == 0:
                continue
            if i >= rest_start[j] and i < rest_start[j] + int(skills.loc[j, 'min_long_rest']):
                sleep.append(skills.loc[j, "Runner"])
        assignment["Sleep"].append(sleep)

    assignment["D+/km"] = [round(assignment["D+"][i] / assignment["Distance"][i]/(1+route.loc[i, "Bike"]), 2) for i in range(len(assignment["D+"]))]
    assignment["D-/km"] = [round(assignment["D-"][i] / assignment["Distance"][i]/(1+route.loc[i, "Bike"]), 2) for i in range(len(assignment["D-"]))]
    assignment_df = pd.DataFrame(assignment)

    rest_block_index = {}
    for j in range(n_people):
        if skills.loc[j, 'min_long_rest'] == 0:
            continue
        for s in range(min_start_rest, n_segments): 
            if (s, j) in r and r[s, j].varValue == 1:
                rest_block_index[j] = s
                break 
    sleep_by_segment = {i: [] for i in segments}
    for j in range(n_people):
        if skills.loc[j, 'min_long_rest'] == 0:
            continue
        if j not in rest_block_index:
            continue
        s_rest = rest_block_index[j]
        s = s_rest - 1
        while s >= 0 and x[s, j].varValue == 0 and d[s, j].varValue == 0:
            sleep_by_segment[segments[s]].append(skills["Runner"][j])
            s -= 1
        if x[s_rest, j].varValue == 0 and d[s_rest, j].varValue == 0:
            sleep_by_segment[segments[s_rest]].append(skills["Runner"][j])
        s = s_rest + 1
        while s < n_segments and x[s, j].varValue == 0 and d[s, j].varValue == 0:
            sleep_by_segment[segments[s]].append(skills["Runner"][j])
            s += 1
    assignment_df["Sleep"] = assignment_df.index.map(lambda i: sleep_by_segment.get(i, []))

    assignment_df.set_index('Segment', inplace=True)
    if save: assignment_df.to_csv('results/schedule.csv')

    # summary activity person
    summary = {
        "Runner": [],
        "Legs": [],
        "Distance": [],
        "D+": [],
        "D-": [],
        "Driving": [],
        "Driving shifts": []
    }
    for j in people:
        assigned_legs = [i for i in segments if pulp.value(x[i, j]) == 1]
        assigned_driving = [i for i in segments if pulp.value(d[i, j]) == 1]
        summary["Runner"].append(skills.loc[j, "Runner"])
        summary["Legs"].append(len(assigned_legs))
        summary["Distance"].append(pulp.value(pulp.lpSum(route.loc[i, "Distance"]/(1+route.loc[i, "Bike"]) for i in assigned_legs)))
        summary["D+"].append(int(pulp.value(pulp.lpSum(route.loc[i, "D+"]/(1+route.loc[i, "Bike"]) for i in assigned_legs))))
        summary["D-"].append(int(pulp.value(pulp.lpSum(route.loc[i, "D-"]/(1+route.loc[i, "Bike"]) for i in assigned_legs))))
        summary["Driving"].append(pulp.value(pulp.lpSum(route.loc[i, "Distance"] for i in assigned_driving)))
        summary["Driving shifts"].append(
            int(sum(pulp.value(z[i, j]) for i in range(1, n_segments) if pulp.value(z[i, j]) == 1))
        )
    summary_df = pd.DataFrame(summary)
    summary_df.set_index('Runner', inplace=True)
    expected_time = prob.objective.value() # in seconds
    expected_time = int(expected_time // 3600), int((expected_time % 3600) // 60), int(expected_time % 60)
    print(f"Expected Time: {expected_time[0]}h{expected_time[1]:02d}m{expected_time[2]:02d}s")
    if save: summary_df.to_csv('results/summary.csv')

    return assignment_df, summary_df

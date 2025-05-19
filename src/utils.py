import pandas as pd 
import ast

def load_route():
    route = pd.read_csv('data/route.csv', skiprows=2)
    route = route[route['valid'] == 1]
    columns = ["Segment", "From", "To", "Distance", "D+", "D-", "night", "Places","Bike"]
    route['Bike'] = route['Bike'].apply(lambda x: 1 if x == 'bike' else 0)
    route = route[columns]
    route['Segment'] = route['Segment'].astype(int)
    route['D+'] = route['D+'].astype(int)
    route['D-'] = route['D-'].astype(int)
    route["night"] = route["night"].astype(int)
    route.reset_index(drop=True, inplace=True)
    return route

def load_skills():
    skills = pd.read_csv('data/skills.csv')
    skills["pace_HM"] = skills["pace_HM"].apply(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))
    return skills

def load_schedule():
    schedule = pd.read_csv('results/schedule.csv', index_col=0).fillna("")
    schedule["Sleep"] = schedule["Sleep"].apply(ast.literal_eval)
    return schedule

def get_basepace(pace_HM, k, distance):
    base_pace = pace_HM * (1 + k * (distance - 21.0975))
    return [f"{int(base_pace_i/60)}:{round((base_pace_i)%60):02d}" for base_pace_i in base_pace]

def get_pace():
    skills = load_skills()
    pace_df = pd.DataFrame(index=skills["Runner"])
    for distance in range(20, 91, 5):
        pace_df[distance] = get_basepace(skills['pace_HM'], skills['endurance'], distance)
    return pace_df
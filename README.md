# EuroTrip - Route Scheduling Solver
Route scheduling solver for EuroTrip 

# Description
This repository is a route scheduling solver for [EuroTrip](https://www.indierunner.nl/events/eurotrip-2025), a running relay event. The race consists in running from a point A to a point B, taking whatever route (just 4 mandatory checkpoints). It is run in team of N poeple (usally 5-8). At least one person has to run at all time and the others are following with a van. A biker to lead the runners along the route is allowed. We formulate here the optimal route scheduling problem as a quadratic programming problem (also considering biking, driving and potentially sleeping assignment). The goal is to minimize the total time spent on the road while satifying all the required constraints and deisderata. We assuming here that the route is already designed and then several internal legs are defined (considering parking constraints).

# Formulation

Objective:
 Minimize the total time spent on the road, i.e., 

 $$\min \sum_{j=1}^{N} t_j$$

where:
<img src="img/equation.png" alt="Objective Explanation" width="400">


Constraints:
1. Necessary Conditions 
    - At least one person has to run at all time, followed by a biker wherever possible.
    - At least one person has to drive at all time (not overlapping with runner/biker).
2. Running Constraints
    - Total running limits per runner (min and max total distance and max D+).
3. Recovery Constraints
    - At least 2 legs of recovery between two legs of running.
    - At least a night sleeping slot per runner with personalized length.
4. Driving Constraints
    - At least 1 leg of recovery after running before driving.
    - Total driving limits per runner.
    - Night driving limits per runner.
    - Bound number of driver changes.


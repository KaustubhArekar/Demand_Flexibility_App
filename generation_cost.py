def generation_cost(flexibility, demand, re, market_price, ppa, battery_energy_capacity, battery_power_capacity, battery_initial_state, solar_pu_cost, wind_pu_cost,market_limit, output_folder_name, num_slots, daily_slots,availability):    
    #RE Power generation profiles
    
    OUTPUT_PATH = 'D:/MpEnsystems/SE4ALL DF 2024 - 2025/DF model/input files for model/output data/'
    
    import time
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum   
    import numpy as np
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    from pulp import LpStatus 
      

    
    start_time = time.time()
    # print(re.sum(axis=1))
    solar=re['Solar']
    wind=re['Wind']
    RE=solar+wind
    
    # Net demand for conventional genration
    net_demand = demand.iloc[:,0] -solar[:]-wind[:]
    # print(net_demand,demand,re.sum(axis=1))

    num_plants=len(ppa['Capacity'])
    technical_minimum  = ppa['Technical_min'].sum()
    min_net_demand = min(net_demand)
    print('Minimum net demand = ', min_net_demand)
    print('Technical minumum - Minimum net demand = ', technical_minimum - min_net_demand)
    print('Technical minumum - Net_demand_start = ',technical_minimum - net_demand[0])
    
    generation_capacity = ppa['Capacity']
    technical_minimum = ppa['Technical_min']
    fixed_cost = ppa['Fixed cost']  # Fixed cost for each plant
    Variable_cost=ppa['Variable cost']
   
  
    
    cost_matrix = pd.DataFrame()

    ramping_up = ppa['Ramping_up'] # Ramping up limit for each plant
    ramping_down = ppa['Ramping down']  # Ramping down limit for each plant
    

    num_plants=len(generation_capacity)
    num_hours=len(demand.values)
#     print(num_hours)
  
# --------------------------------------------------------------------------------

    problem = LpProblem("Power_Generation_Optimization", LpMinimize)

    # Define the decision variables
    schedule = [[LpVariable(f"Schedule_{t}_{p}") for p in range(num_plants)] 
                for t in range(num_hours)]
    del_schedule = [[LpVariable(f"del_Schedule_{t}_{p}") for p in range(num_plants)] 
                for t in range(num_hours)]
    market_drawl = [LpVariable(f"Market_{t}") for t in range(num_hours)]
    optimised_demand = [LpVariable(f"opt_demand_{t}") for t in range(num_hours)]
    del_optimised_demand = [LpVariable(f"del_opt_demand_{t}") for t in range(num_hours)]
    
    b_soc = [LpVariable(f"b_soc_{t}",lowBound=0.10*battery_energy_capacity, upBound = battery_energy_capacity) for t in range(num_hours+1)]
    b_power = [LpVariable(f"b_power_{t}",lowBound=-battery_power_capacity, upBound = battery_power_capacity) for t in range(num_hours)]
    total_shift = [LpVariable(f"{t}_total_shift") for t in range(num_hours)] 
    availability_v = [[LpVariable(f"availability_{t}_{p}", cat='Binary') for p in range(num_plants)] for t in range(num_hours)] 
    # set initial battery state of charge
    problem += b_soc[0] == battery_initial_state
               
   
    
    # Set the objective function
    problem += lpSum(schedule[t][p] * Variable_cost[p] for t in range(num_hours) for p in range(num_plants))+lpSum(market_drawl[t]*market_price['rate'][t] for t in range(num_hours))

    
    # Add the constraints

    #Constraint 1  - ramping up and down constraints 1% for thermal, 3% for gas, 10% for hydro
    for t in range(num_hours - 1):
        for p in range(num_plants):
            problem+= del_schedule[t][p] == schedule[t+1][p] - schedule[t][p]
            problem += del_schedule[t][p] >= ramping_down[p]  
            problem +=  del_schedule[t][p]<= ramping_up[p] 
    
        

    # Constraint 2 -  technical Minimum
    for t in range(num_hours):
        for p in range(num_plants):
            problem += schedule[t][p]>=technical_minimum[p]*availability_v[t][p]



    #Constraint 3 - Total generation in slot i = Total demand in slot i - IEX market power - Battery discharge        
    for t in range(num_hours):
        problem += lpSum(schedule[t][p] for p in range(num_plants)) ==optimised_demand[t]-re.iloc[t,:].sum()-market_drawl[t]-b_soc[t]+b_soc[t+1]
        
    #Constraint 4 - Generation should not exceed capacity
    for t in range(num_hours):
        for p in range(num_plants):
            problem += schedule[t][p] <= ppa['Capacity'][p]*availability_v[t][p]
   
            
    #constraint 5 - sales in market power should be less than market limit set 
    for t in range(num_hours):
        problem +=market_drawl[t] >= -1*market_limit
    
    for t in range(num_hours):
        problem +=market_drawl[t] <= 1*market_limit
        
        
    #constraint 6 - battery charging and discharging constriants

    for t in range(num_hours):
        problem +=b_power[t] ==b_soc[t+1]-b_soc[t]

    #constraint 7 - Optimised demand should be equal to net demand after DR and DF
    for t in range(num_hours):
        problem += optimised_demand[t] <= (1+flexibility)*demand.iloc[t,0]
        problem += optimised_demand[t] >= (1-flexibility)*demand.iloc[t,0]

    #constraint 8 - total shift for calculating inconvinience cost
    for t in range(num_hours):
        problem += total_shift[t]>=demand.iloc[t,0] - optimised_demand[t]
        problem += total_shift[t]>=-demand.iloc[t,0] + optimised_demand[t]

    

    num_days = 365
    for d in range(num_days):
            problem += lpSum(optimised_demand[t+d*daily_slots] for t in range(daily_slots)) ==demand[d*daily_slots:(d+1)*daily_slots].sum()
            problem += lpSum(total_shift[t+d*daily_slots] for t in range(daily_slots)) <= 0.005*demand[d*daily_slots:(d+1)*daily_slots].sum()
    # Solve the problem
    problem.solve()

    # Check the status of the solution
    
    print("Generation optimisation Status:", LpStatus[problem.status])  


    end_time0=time.time()
 #______________________________________________________________________________________________________   
    # saving variable values in dataframe

    schedule_gen = pd.DataFrame(index=range(num_hours), columns=range(num_plants))
    market_power = pd.DataFrame(index=range(num_hours))
    battery_profile = pd.DataFrame(index=range(num_hours))
    batt_soc=pd.DataFrame(index=range(num_hours))
    opt_demand=pd.DataFrame(index=range(num_hours))
   
   
    for t in range(num_hours):
        for p in range(num_plants):
            schedule_gen.at[t, p] = schedule[t][p].varValue
        market_power.at[t,0] = market_drawl[t].varValue
        batt_soc.at[t,0]=b_soc[t+1].varValue
        opt_demand.at[t,0]=optimised_demand[t].varValue

        
    for t in range(num_hours):
        battery_profile.at[t,0]=b_power[t].varValue
    
    battery_energy_cost = -1.3*battery_profile*(battery_profile<0)
    
    # dataframe to csv
    gen_schedule_path = os.path.join(OUTPUT_PATH,output_folder_name,"Generation schedules")
    n=0
    if n==0:
        os.makedirs(gen_schedule_path, exist_ok=True)
        
    # schedule_gen.to_csv(gen_schedule_path + "/schedule_output_" + str(n) + ".csv", index=False)
    
    # generation cost = schedule of plant * variable_cost
    re_cost = solar*solar_pu_cost + wind*wind_pu_cost
    market_cost = market_power.iloc[:,0]*market_price.iloc[:,0]
  
    # total_generation_cost = np.dot(np.array(schedule_gen.iloc[:,:]),np.array(Variable_cost[:])) + re_cost +market_cost
    total_generation_cost = np.dot(np.array(schedule_gen.iloc[:,:]),np.array(Variable_cost[:])) +market_cost+battery_energy_cost.iloc[:,0]
    
    sch_gen = schedule_gen.iloc[:,:].sum(axis=1)
 
    per_unit_generation_cost = total_generation_cost/(opt_demand.iloc[:,0]+market_power.iloc[:,0])
    # per_unit_generation_cost = total_generation_cost/(projected_demand['demand'])
    end_time= time.time()
    
    total_time1 = end_time0 - start_time
    total_time2 = end_time - start_time
#     print('Time taken for optimisation:',total_time1)
    print('Total time elapsed for generation optimisation:',total_time2)

    

    return total_generation_cost,per_unit_generation_cost, demand , opt_demand, net_demand,schedule_gen, market_power, battery_profile

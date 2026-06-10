def daily_generation_cost(demand, re, ppa, output_folder_name, num_slots, daily_slots):    
    #RE Power generation profiles
    
    OUTPUT_PATH = 'D:/MpEnsystems/SE4ALL DF 2024 - 2025/DF model/input files for model/output data/'
    
    import time
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus  
    import numpy as np
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
      

    
    start_time = time.time()
    solar=re['Solar']
    wind=re['Wind']
    RE=solar+wind
    
    # Net demand for conventional genration
    net_demand = demand.iloc[:,0] -solar[:]-wind[:]

    #daily demand requirement
    daily_demand = pd.DataFrame(index=range(len(demand)//daily_slots), columns=['demand'])
    daily_re=pd.DataFrame(index=range(len(demand)//daily_slots), columns=['RE'])
    for i in range(len(demand)//daily_slots):
        daily_demand.iloc[i,0]=sum(net_demand.iloc[i*daily_slots:(i+1)*daily_slots])
        daily_re.iloc[i,0]=(re.iloc[i*daily_slots:(i+1)*daily_slots,:].sum().sum())

    print(daily_demand, daily_re)

    num_plants=len(ppa['Capacity'])
    technical_minimum  = ppa['Technical_min'].sum()
    min_net_demand = min(net_demand)
    print('Minimum net demand = ', min_net_demand)
    print('Technical minumum - Minimum net demand = ', technical_minimum - min_net_demand)
    print('Technical minumum - Net_demand_start = ',technical_minimum - net_demand[0])
    
    generation_capacity = ppa['Capacity']
    technical_minimum = ppa['Technical_min']
    Variable_cost=ppa['Variable cost']
   

    ramping_up = 24*(daily_slots/24)*ppa['Ramping_up'] # Ramping up limit for each plant
    ramping_down = 24*(daily_slots/24)*ppa['Ramping down']  # Ramping down limit for each plant


    num_plants=len(generation_capacity)
    num_hours=len(demand.values)
#     print(num_hours)
  
# --------------------------------------------------------------------------------

    problem = LpProblem("Power_Generation_Optimization", LpMinimize)

    # Define the decision variables
    schedule = [[LpVariable(f"Schedule_{t}_{p}") for p in range(num_plants)] 
                for t in range(num_hours//daily_slots)]
    del_schedule = [[LpVariable(f"del_Schedule_{t}_{p}") for p in range(num_plants)] 
                for t in range(num_hours//daily_slots)]
   
    availability_v = [[LpVariable(f"availability_{t}_{p}", cat='Binary') for p in range(num_plants)] for t in range(num_hours//daily_slots)] 
    
               
    # Set the objective function
    problem += lpSum(schedule[t][p] * Variable_cost[p] for t in range(num_hours//daily_slots) for p in range(num_plants))

    
    # Add the constraints

    #Constraint 1  - ramping up and down constraints 1% for thermal, 3% for gas, 10% for hydro
    for t in range((num_hours//daily_slots) - 1):
        for p in range(num_plants):
            problem+= del_schedule[t][p] == schedule[t+1][p] - schedule[t][p]
            problem += del_schedule[t][p] >= ramping_down[p]  
            problem +=  del_schedule[t][p]<= ramping_up[p] 
    
        

    # Constraint 2 -  technical Minimum
    for t in range(num_hours//daily_slots):
        for p in range(num_plants):
            problem += schedule[t][p]>=technical_minimum[p]*availability_v[t][p]*24



    #Constraint 3 - Total generation in slot i = Total demand in slot i - IEX market power - Battery discharge        
    for t in range(num_hours//daily_slots):
        problem += lpSum(schedule[t][p] for p in range(num_plants)) ==daily_demand.iloc[t,0]-daily_re.iloc[t,0]
        
    #Constraint 4 - Generation should not exceed capacity
    for t in range(num_hours//daily_slots):
        for p in range(num_plants):
            problem += schedule[t][p] <= 24*ppa['Capacity'][p]*availability_v[t][p]
   
    
    problem.solve()
    print("Generation optimisation Status:", LpStatus[problem.status])  


    end_time0=time.time()
 #______________________________________________________________________________________________________   
    # saving variable values in dataframe

    schedule_gen = pd.DataFrame(index=range(num_hours//daily_slots), columns=range(num_plants))
    availability=pd.DataFrame(index=range(num_hours//daily_slots), columns=range(num_plants))

   
   
    for t in range(num_hours//daily_slots):
        for p in range(num_plants):
            schedule_gen.at[t, p] = schedule[t][p].varValue
            availability.at[t,p] = availability_v[t][p].varValue

    # dataframe to csv
    gen_schedule_path = os.path.join(OUTPUT_PATH,output_folder_name,"Generation schedules")
    availability_path = os.path.join(OUTPUT_PATH,output_folder_name,"Generation schedules")
    
    # os.makedirs(gen_schedule_path, exist_ok=True)
        
    schedule_gen.to_csv(gen_schedule_path + "/schedule_output_"  + ".csv", index=False)
    availability.to_csv(availability_path + "/availability_output_" + ".csv", index=False)
        

    return net_demand,schedule_gen, availability


INPUT_PATH = 'D:/MpEnsystems/SE4ALL DF 2024 - 2025/DF model/input files for model/input data 2 - Copy/'
OUTPUT_PATH = 'D:/MpEnsystems/SE4ALL DF 2024 - 2025/DF model/input files for model/output/'
path_for_meter_data = 'D:/MpEnsystems/SE4ALL DF 2024 - 2025/DF model/input files for model/input data/meter data/'

import ast
import pandas as pd     
import os


ppa =pd.read_excel(INPUT_PATH + 'Generation_stack.xlsx') # PPA data of thermal plants (Capacity, ramping limts, technical min, variable cost)
print('Succuefully read: Generation_stack.xlsx')
projected_demand = pd.read_csv(INPUT_PATH + 'demand.csv') # Discom hourly demand in MW 
print('Succuefully read: demand.csv')
re = pd.read_excel(INPUT_PATH +'RE - high.xlsx') 
print('Succuefully read: RE - high.xlsx')
input_parameters = pd.read_csv(INPUT_PATH +'input_parameters.csv') # User defined parameters
print('Succuefully read: input_parameters.csv')

demand = projected_demand['demand']

num_slots = 24

daily_slots = int(float(input_parameters[input_parameters['Parameter']=='daily_slots']['Value'].values[0]))
output_folder_name =(input_parameters[input_parameters['Parameter']=='Output Folder Name']['Value'].values[0])


[net_demand,schedule_gen, availability]=daily_generation_cost(projected_demand, re, ppa, output_folder_name, num_slots, daily_slots)
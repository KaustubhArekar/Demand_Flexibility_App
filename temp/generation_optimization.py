def generation_optimization(flexibility, demand, re, market_price, ppa, battery_energy_capacity, battery_power_capacity, battery_initial_state, solar_pu_cost, wind_pu_cost,market_limit, output_folder_name, num_slots, daily_slots):    

    
    OUTPUT_PATH = 'D:/MpEnsystems/SE4ALL DF 2024 - 2025/DF model/input files for model/output data/'
    
    import time
    import pandas as pd
    import matplotlib.pyplot as plt
    from pyomo import environ as pyo
   

    print('Starting generation optimization...')
    start_time = time.time()
    # print(re.sum(axis=1))
    solar=re['Solar']
    wind=re['Wind']
    RE=solar+wind
    
    # Net demand for conventional genration
    net_demand = demand.iloc[:] -solar[:]-wind[:]
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
    market_rate = market_price['rate']
   
    num_days = int(len(demand.values)/daily_slots)
    
    cost_matrix = pd.DataFrame()

    ramping_up = ppa['Ramping_up'] # Ramping up limit for each plant
    ramping_down = ppa['Ramping down']  # Ramping down limit for each plant
    


    num_plants=len(generation_capacity)
    num_hours=len(demand.values)

    availability_dict = {}
    for t in range(num_hours):
        for i in range(num_plants):
            availability_dict[(t, i)] = availability.iloc[t, i]
#     print(num_hours)
  
# --------------------------------------------------------------------------------

    model = pyo.ConcreteModel()

    # Define decision variables
    model.schedule = pyo.Var(range(num_hours), range(num_plants), domain=pyo.NonNegativeReals)  # Generation from each plant at each hour
    model.del_schedule = pyo.Var(range(num_hours), range(num_plants), domain=pyo.Reals)  # Generation from each plant at each hour
    model.battery_power = pyo.Var(range(num_hours))  # Battery power at each hour
    model.battery_soc = pyo.Var(range(num_hours), domain=pyo.NonNegativeReals)  # State of charge of the battery at each hour
    model.market_power = pyo.Var(range(num_hours))  # Power bought from the market at each hour        
    model.opt_demand = pyo.Var(range(num_hours), domain=pyo.NonNegativeReals)  # Optimized demand at each hour
    model.battery_soc[0].fix(battery_initial_state)  # Initial state of charge of the battery
    model.shifted_demand = pyo.Var(range(num_hours))  # Shifted demand at each hour
    model.daily_shifted_demand = pyo.Var(range(num_days))  # Daily shifted demand for each slot
    model.daily_demand = pyo.Var(range(num_days))  # Daily demand for each slot
    model.availability = pyo.Var(range(num_hours), range(num_plants))  # Availability of each plant at each hour

    # Define objective function

    def availability_rule(model, t, i):
        return model.availability[t, i] <=1
    def objective_rule(model):
        generation_cost = sum(Variable_cost[i] * model.schedule[t, i] for i in range(num_plants) for t in range(num_hours))
        market_cost = sum(market_rate[t] * model.market_power[t] for t in range(num_hours))
        return generation_cost + market_cost 
    
    #define constraints
    def demand_constraint_rule(model, t):
        return sum(model.schedule[t, i] for i in range(num_plants)) + model.battery_power[t] + model.market_power[t] == model.opt_demand[t] - solar[t] - wind[t]       
    
    def technical_minimum_rule(model, t, i):
        return model.schedule[t, i] >= technical_minimum[i]*model.availability[t, i]

    def generation_capacity_rule(model, t, i):
        return model.schedule[t, i] <= generation_capacity[i]*model.availability[t, i]
    
    def del_schedule_rule(model, t,i):
        if t == 0:
            return pyo.Constraint.Skip  # No ramping constraint for the first hour
        return model.del_schedule[t, i] == model.schedule[t, i] - model.schedule[t-1, i]
    
    def ramping_up_rule(model,t,i):
        if t == 0:
            return pyo.Constraint.Skip  # No ramping constraint for the first hour
        return model.del_schedule[t, i] <= ramping_up[i]
    
    def ramping_down_rule(model, t, i):
        if t == 0:
            return pyo.Constraint.Skip  # No ramping constraint for the first hour
        return model.del_schedule[t, i] >= ramping_down[i]
    
    def battery_soc_rule(model, t):
        if t == 0:
            return pyo.Constraint.Skip  # Initial state of charge is fixed
        return model.battery_soc[t] == model.battery_soc[t-1] - model.battery_power[t]
    

    def market_limit_rule(model, t):
        return model.market_power[t] <= market_limit
    
    def flexibility_rule_p(model, t):
        return model.opt_demand[t] >= demand.values[t]*(1+flexibility)
    
    def flexibility_rule_n(model, t):
        return model.opt_demand[t] <= demand.values[t]*(1-flexibility)

    def flexibility_rule(model, t):
        return model.shifted_demand[t] == (demand.values[t] - model.opt_demand[t])
    
    def shifted_demand_rule(model, t):
        return model.daily_shifted_demand[t] == sum(model.shifted_demand[i] for i in range((t)*daily_slots,(t+1)*daily_slots))

    def daily_demand_rule(model, t):
        return model.daily_demand[t] == sum(model.opt_demand[i] for i in range((t)*daily_slots,(t+1)*daily_slots))
    
    def flexibility_limit_rule(model, t):
        return model.daily_shifted_demand[t] <= model.daily_demand[t]*flexibility

    def battery_power_limit_rule(model, t):
        return model.battery_power[t] <= battery_power_capacity
    
    def batteery_energy_limit_rule(model, t):
        return model.battery_soc[t] <= battery_energy_capacity
    
    
    

    # Add constraints to the model
    model.demand_constraint = pyo.Constraint(range(num_hours), rule=demand_constraint_rule)
    model.availability_constraint = pyo.Constraint(range(num_hours), range(num_plants), rule=availability_rule)
    model.technical_minimum_constraint = pyo.Constraint(range(num_hours), range(num_plants), rule=technical_minimum_rule)
    model.generation_capacity_constraint = pyo.Constraint(range(num_hours), range(num_plants), rule=generation_capacity_rule)
    model.ramping_up_constraint = pyo.Constraint(range(num_hours), range(num_plants), rule=ramping_up_rule)
    model.ramping_down_constraint = pyo.Constraint(range(num_hours), range(num_plants), rule=ramping_down_rule)
    model.battery_soc_constraint = pyo.Constraint(range(num_hours), rule=battery_soc_rule)
    model.market_limit_constraint = pyo.Constraint(range(num_hours), rule=market_limit_rule)
    model.flexibility_constraint = pyo.Constraint(range(num_hours), rule=flexibility_rule)
    model.flexibility_limit_constraint = pyo.Constraint(range(num_days), rule=flexibility_limit_rule)   
    model.flexibility_rule_p_constraint = pyo.Constraint(range(num_hours), rule=flexibility_rule_p)
    model.flexibility_rule_n_constraint = pyo.Constraint(range(num_hours), rule=flexibility_rule_n)
    model.battery_power_limit_constraint = pyo.Constraint(range(num_hours), rule=battery_power_limit_rule)
    model.battery_energy_limit_constraint = pyo.Constraint(range(num_hours), rule=batteery_energy_limit_rule)
    model.shifted_demand_constraint = pyo.Constraint(range(num_days), rule=shifted_demand_rule)
    model.daily_demand_constraint = pyo.Constraint(range(num_days), rule=daily_demand_rule) 
    model.del_schedule_constraint = pyo.Constraint(range(num_hours), range(num_plants), rule=del_schedule_rule)


    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Solve the model
    # solver = pyo.SolverFactory('cbc')
    # solver = MindtPySolver()
    result = pyo.SolverFactory('mindtpy').solve(
    model, 
    strategy='OA',
    mip_solver='cbc',
    nlp_solver='ipopt',
   
    # mip_solver_args={'ratio': 0.05, 'sec': 300}, # 5% gap or 5 mins max per master problem
    # mip_solver_tee=True,                         # Show us what CBC is doing!
    # nlp_solver_tee=False,
    # ------------------------------
    tee=True
)

    print(result)
    print('Generation optimization completed.')

    # Extract results
    total_generation_cost = pyo.value(model.objective)
    per_unit_generation_cost = total_generation_cost / sum(model.schedule[t, i].value for i in range(num_plants) for t in range(num_hours))
    opt_demand = [model.schedule[t, i].value for i in range(num_plants) for t in range(num_hours)]
    net_demand = [net_demand.values[t] for t in range(num_hours)]           
    schedule_gen = [[model.schedule[t, i].value for i in range(num_plants)] for t in range(num_hours)]
    market_power = [model.market_power[t].value for t in range(num_hours)]
    battery_profile = [model.battery_soc[t].value for t in range(num_hours)]

    schedule_gen_dict = {}
    for t in range(num_hours):
        for i in range(num_plants):
            schedule_gen_dict[(t, i)] = model.schedule[t, i].value

    schedule_gen_df = pd.DataFrame(schedule_gen, columns=[f'Plant_{i}' for i in range(num_plants)])
    schedule_gen_df.to_csv(OUTPUT_PATH + f'{output_folder_name}/schedule_gen.csv', index=False)

    print(schedule_gen_dict)
    return total_generation_cost,per_unit_generation_cost, demand , opt_demand, net_demand,schedule_gen, market_power, battery_profile


import ast
import pandas as pd
import os
import matplotlib.pyplot as plt

# read input data
INPUT_PATH = 'D:/MpEnsystems/SE4ALL DF 2024 - 2025/DF model/input files for model/input data 2 - Copy/'
OUTPUT_PATH = 'D:/MpEnsystems/SE4ALL DF 2024 - 2025/DF model/input files for model/output/'
path_for_meter_data = 'D:/MpEnsystems/SE4ALL DF 2024 - 2025/DF model/input files for model/input data/meter data/'


ppa =pd.read_excel(INPUT_PATH + 'Generation_stack.xlsx') # PPA data of thermal plants (Capacity, ramping limts, technical min, variable cost)
print('Succuefully read: Generation_stack.xlsx')
projected_demand = pd.read_csv(INPUT_PATH + 'demand.csv') # Discom hourly demand in MW 
print('Succuefully read: demand.csv')
market_price = pd.read_excel(INPUT_PATH+'market_rate.xlsx') # Power market RTM prices for last year
print('Succuefully read: market_rate.xlsx')
re = pd.read_excel(INPUT_PATH +'RE - high.xlsx') 
print('Succuefully read: RE - high.xlsx')
tariff = pd.read_csv(INPUT_PATH +'tariff.csv') # cluster wise hourly tariff
print('Succuefully read: tariff.csv')
input_parameters = pd.read_csv(INPUT_PATH +'input_parameters.csv') # User defined parameters
print('Succuefully read: input_parameters.csv')
availability = pd.read_excel(INPUT_PATH +'availability.xlsx') # availability of thermal plants
print('Succuefully read: availability.xlsx')

num_slots = 24
flex =ast.literal_eval(input_parameters[input_parameters['Parameter']=='flexibility']['Value'].values[0])
base_incenitve_DF =float(input_parameters[input_parameters['Parameter']=='base_incenitve_DF']['Value'].values[0])
base_incenitve_DR = float(input_parameters[input_parameters['Parameter']=='base_incenitve_DR']['Value'].values[0])
DR_lambda = float(input_parameters[input_parameters['Parameter']=='DR_lambda']['Value'].values[0])
DF_lambda = float(input_parameters[input_parameters['Parameter']=='DF_lambda']['Value'].values[0])
daily_slots = int(float(input_parameters[input_parameters['Parameter']=='daily_slots']['Value'].values[0]))
step_size = float(input_parameters[input_parameters['Parameter']=='step_size']['Value'].values[0])
solar_pu_cost =float(input_parameters[input_parameters['Parameter']=='solar_pu_cost']['Value'].values[0])
wind_pu_cost =float(input_parameters[input_parameters['Parameter']=='wind_pu_cost']['Value'].values[0])
market_limit = float(input_parameters[input_parameters['Parameter']=='market_limit']['Value'].values[0])
battery_power_capacity =float(input_parameters[input_parameters['Parameter']=='battery_power_capacity']['Value'].values[0])
battery_energy_capacity =float(input_parameters[input_parameters['Parameter']=='battery_energy_capacity']['Value'].values[0])
battery_initial_state = float(input_parameters[input_parameters['Parameter']=='battery_initial_state']['Value'].values[0])
clusters_rqd = ast.literal_eval(input_parameters[input_parameters['Parameter']=='clusters_rqd']['Value'].values[0])
mode=ast.literal_eval(input_parameters[input_parameters['Parameter']=='mode']['Value'].values[0])
inconvenience_cost =ast.literal_eval(input_parameters[input_parameters['Parameter']=='inconvenience_cost']['Value'].values[0])
output_folder_name =(input_parameters[input_parameters['Parameter']=='Output Folder Name']['Value'].values[0])
Re_forecast =ast.literal_eval(input_parameters[input_parameters['Parameter']=='RE_forecasting']['Value'].values[0])



total_generation_cost,per_unit_generation_cost, demand , opt_demand, net_demand,schedule_gen, market_power, battery_profile = generation_optimization(flex[0], projected_demand['demand'], re, market_price, ppa, battery_energy_capacity, battery_power_capacity, battery_initial_state, solar_pu_cost, wind_pu_cost,market_limit, output_folder_name, num_slots, daily_slots)

# print('schedule generation: ' + schedule_gen)
# plt.plot(total_generation_cost, label='Total Generation Cost')
# plt.xlabel('Time')
# plt.ylabel('Cost')
# plt.title('Generation Cost Over Time')
# plt.legend()
# plt.show()

# plt.plot(schedule_gen.iloc[:,1], label='Plant 1 Generation')
# plt.xlabel('Time')
# plt.ylabel('Cost')
# plt.title('Plant 1 Generation Over Time')
# plt.legend()
# plt.show()

print('Total generation cost: ', total_generation_cost)

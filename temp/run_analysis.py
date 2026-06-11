from generation_cost import generation_cost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from read_data import read_data
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go

ppa, projected_demand, market_price, re, tariff, flex, base_incenitve_DF, base_incenitve_DR, DR_lambda, DF_lambda, daily_slots, step_size, solar_pu_cost, wind_pu_cost, market_limit, battery_power_capacity, battery_energy_capacity, battery_initial_state, clusters_rqd, mode,inconvenience_cost, Re_forecast, INPUT_PATH, OUTPUT_PATH, output_folder_name = read_data()
    
demand = projected_demand['demand']

num_slots = 8760
flexibility = [0,0.1,0.2]
pu_gen_cost_flex = pd.DataFrame(index = range(num_slots), columns = [flexibility])    
total_cost_flex = pd.DataFrame(index = range(num_slots), columns = [flexibility])   
total_cost_sum = pd.DataFrame(index = range(len(flexibility)), columns = ['total_cost_sum']) 
for flex in flexibility:
    print('Flexibility = ', flex)
    [total_cost,pu_gen_cost, demand, opt_demand, net_demand] = generation_cost(flex, demand, re, market_price, ppa, battery_energy_capacity, battery_power_capacity, battery_initial_state, solar_pu_cost, wind_pu_cost)
    pu_gen_cost_flex[flex] = pu_gen_cost
    total_cost_flex[flex] = total_cost
    total_cost_sum[flex] = total_cost.sum()


for flex in range(len(flexibility)):
    plt.plot(pu_gen_cost_flex.iloc[0:96,flex], label = 'Flexibility = '+str(flex))
    plt.xlabel('Time (hours)')
    plt.ylabel('Per unit generation cost (Rs./kWh)')

plt.legend()
plt.show()

for flex in range(len(flexibility)):
    plt.plot(pu_gen_cost_flex.iloc[:,flex]-pu_gen_cost_flex.iloc[:,0], label = 'Flexibility = '+str(flex))
    plt.xlabel('Time (hours)')
    plt.ylabel('Per unit generation cost (Rs./kWh)')

plt.legend()
plt.show()


for flex in range(len(flexibility)):
    plt.plot(total_cost_flex.iloc[0:96,flex], label = 'Flexibility = '+str(flex))
    plt.xlabel('Time (hours)')
    plt.ylabel('Total generation cost (Rs.)')

plt.legend()
plt.show()


plt.plot(total_cost_sum[:], label = 'Total generation cost for different flexibility levels')
plt.xlabel('Flexibility level') 
plt.ylabel('Total generation cost (Rs.)')
plt.legend()
plt.show()


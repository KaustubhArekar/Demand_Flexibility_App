# if Re_forecast ==1:
#     re['Solar']=final_SolarPower['Total_power']
#     re['Wind']=final_WindPower['Total_power']

import time

from read_data import read_data
from load_profile_clustering import load_profile_clustering
from trigger_calculation import trigger_calculation
from compute_generation_cost import compute_generation_cost
from load_optimisation_aggregate import load_optimisation_aggregate 
from load_optimisation import load_optimisation 
from pulp import LpStatus

ppa, projected_demand, market_price, re, tariff, flex, base_incenitve_DF, base_incenitve_DR, DR_lambda, DF_lambda, daily_slots, step_size, solar_pu_cost, wind_pu_cost, market_limit, battery_power_capacity, battery_energy_capacity, battery_initial_state, clusters_rqd, mode,inconvenience_cost, output_folder_name, Re_forecast, output_folder_name, path_for_meter_data ,path_for_meter_data= read_data()

start_time = time.time()
# create dictionaries to store data in each iteration
iterative_tariff_signals ={}
iterative_demand ={}
iterative_pu_gen_cost = {}
iterative_opt_load = {}
iterative_cluster_bills = {}
iterative_market_power = {}
iterative_battery_dispatch ={}
iterative_total_gen_cost = {}

# initial conditions BAU without DF and DR
# [consumer_load_clusters, aggregate_demand_from_clusters,cluster_bills,min_indexes,tarifff] =load_profile_clustering(tariff, path_for_meter_data)
# print(800*consumer_load_clusters[0:24])

n= 0 
print('iteration: ',n)
iterative_tariff_signals[n] = tarifff
iterative_demand[n]= projected_demand['demand']
print('Generation optimisation....')
[pu_gen_cost, market_power, battery_dispatch, total_generation_cost] = compute_generation_cost(iterative_demand[0],n)
iterative_pu_gen_cost[n] = pu_gen_cost

# scaling cluster demand by factor of 800
consumer_load_clusters_scaled = consumer_load_clusters*800
aggregate_demand_from_clusters_scaled =consumer_load_clusters_scaled.sum(axis=1)
cluster_bills_scaled = cluster_bills*800000
iterative_opt_load[n] = consumer_load_clusters_scaled
iterative_market_power[n]=market_power
iterative_battery_dispatch[n] = battery_dispatch
iterative_cluster_bills [n] = cluster_bills_scaled
iterative_total_gen_cost[n] = 1000*(total_generation_cost.sum())

fixed_demand = iterative_demand[0] - aggregate_demand_from_clusters_scaled

print(iterative_total_gen_cost[n]/1000000000)
print(iterative_cluster_bills[n].sum().sum()/1000000000)

print('---------------------------------')
# Sequesntial process of determing incentive struture by DISCOM and DF/DR response from consumers
gen_cost_i = pu_gen_cost
# max number of iterations
number_of_iterations =10
n=n+1
while n < number_of_iterations:
    print('iteration: ',n)
    # Incentive / tariff signal estimation
    
    print('calculating triggers....')
    cluster_triggers = trigger_calculation (iterative_pu_gen_cost[0], tarifff,mode,n)
    
    # Introduce flexibility
    print('Load curves optimisations....')
    optimised_cluster_loads, cluster_bills_i = load_optimisation_aggregate (cluster_triggers,mode,flex,consumer_load_clusters_scaled,tarifff,inconvenience_cost)
    aggregated_opt_cl_load = optimised_cluster_loads.sum(axis = 1)
    
    # calculate net_demand
    demand_i = fixed_demand + aggregated_opt_cl_load
    net_demand_i  = demand_i
    print('Generation optimisation....')
    
    # Comput updated gen cost 
    [pu_gen_cost_i, market_power_i, battery_dispatch_i,total_generation_cost_i] = compute_generation_cost(net_demand_i,n)
    
    gen_cost_i = pu_gen_cost_i
    
   
    iterative_tariff_signals[n] = cluster_triggers
    iterative_demand[n]= demand_i
    iterative_pu_gen_cost[n] = pu_gen_cost_i
    iterative_opt_load[n] = optimised_cluster_loads
    iterative_market_power[n] = market_power_i
    iterative_battery_dispatch[n] = battery_dispatch_i
    iterative_cluster_bills [n] = cluster_bills_i
    iterative_total_gen_cost[n] =1000*(total_generation_cost_i.sum())
     
    
    change_in_DF_consumer_bill = iterative_cluster_bills[n-1].sum() - iterative_cluster_bills[n].sum() # should be positive for consumer benefit
    change_in_profit_to_utility= (iterative_total_gen_cost[n-1] - iterative_total_gen_cost[n] - change_in_DF_consumer_bill.sum()) #should be positive for utility benefit
#     print(iterative_total_gen_cost[n]/1000000000)
#     print(iterative_cluster_bills[n].sum().sum()/1000000000)
    print('---------------------------------')
    
    net_profit_n_1 =-1*((iterative_total_gen_cost[n -1] -iterative_total_gen_cost[0]) - (iterative_cluster_bills[n-1].sum().sum() - iterative_cluster_bills[0].sum().sum()))
    net_profit_n = -1*((iterative_total_gen_cost[n] -iterative_total_gen_cost[0]) - (iterative_cluster_bills[n].sum().sum() - iterative_cluster_bills[0].sum().sum()))

    print(net_profit_n_1, net_profit_n, n)
    
    # check net_profite trend and stop the iterations
    if n>1 and net_profit_n < net_profit_n_1:
        total_iterations = n+1
        break
    elif iterative_cluster_bills[n].sum().sum() > iterative_cluster_bills[0].sum().sum():
        total_iterations = n+1
        break
        
    n=n+1
    
    
    # Battery parametric simulations (optional)
#     battery_power_capacity = battery_power_capacity +500
#     battery_energy_capacity = battery_power_capacity*2*(n)
#     battery_initial_state = battery_energy_capacity*0.5
    total_iterations =n
end_time = time.time()

print(end_time - start_time)
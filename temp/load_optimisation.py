def load_optimisation(triggers, clustered_profile, flexibility, mode,inconvenience_cost):
    
    num_hours = len(clustered_profile)
#     daily_slots = 24
    num_days = int(num_hours/daily_slots)

    if mode == 'DF':
        Bill_optimization = LpProblem("Bill_optimization", LpMinimize)
   
        shifted_load = [LpVariable(f"{t}_shifted_load", lowBound=clustered_profile[t]*(1-flexibility),upBound = clustered_profile[t]*(1+flexibility)) for t in range(num_hours)]

        total_shift = [LpVariable(f"{t}_total_shift") for t in range(num_hours)]        
   
      # Constraint 1 Total energy consumption is same for the day
        for d in range(num_days):
            Bill_optimization += lpSum(shifted_load[t+d*daily_slots] for t in range(daily_slots)) == clustered_profile[d*daily_slots:(d+1)*daily_slots].sum()
    
      # Constraint 2 total shifted energy for calculating inconvinence
        for t in range(num_hours):
                   Bill_optimization += total_shift[t]>=clustered_profile[t] - shifted_load[t]
                   Bill_optimization += total_shift[t]>=-clustered_profile[t] + shifted_load[t]
       
        
#      # inconvinience to consumer  
        incovinience = lpSum(total_shift[t]*inconvenience_cost for t in range(num_hours))
    
        #Objective fucntcion (minimise the bill i.e. electricity cost to consumers + inconvinence)
        Bill_optimization += lpSum(shifted_load[t] * triggers[t] for t in range(num_hours)) +incovinience                         
        
        
        Bill_optimization.solve()

    
        opt_load = pd.DataFrame(index=range(len(shifted_load)), columns=range(1))
        
    elif mode == 'DR':
    
        Bill_optimization = LpProblem("Bill_optimization", LpMinimize)
      
        shifted_load = [LpVariable(f"{t}_shifted_load", lowBound=clustered_profile[t]*(1-flexibility),upBound = clustered_profile[t]) for t in range(num_hours)]
        total_shift = lpSum(clustered_profile[t] -shifted_load[t] for t in range(num_hours))

      # Objective fucntcion (minimise the bill i.e. electricity cost to consumers + inconvinence)
        Bill_optimization += lpSum(shifted_load[t] *triggers[t] for t in range(num_hours)) +total_shift*inconvenience_cost
   
  
      # Constraint 1 Total energy reduction limitted to 5 % of total baseline consumption
        for d in range(num_days):
            Bill_optimization += lpSum(shifted_load[t+d*daily_slots] for t in range(daily_slots)) >= (1-0.05)*clustered_profile[d*daily_slots:(d+1)*daily_slots].sum()
    
        Bill_optimization.solve()

        opt_load = pd.DataFrame(index=range(len(shifted_load)), columns=range(1))
        
    else:
            
        Bill_optimization = LpProblem("Bill_optimization", LpMinimize)
    
        shifted_load = [LpVariable(f"{t}_shifted_load", lowBound=clustered_profile[t]*(1-flexibility),upBound = clustered_profile[t]) for t in range(num_hours)]
        total_shift = lpSum(clustered_profile[t] -shifted_load[t] for t in range(num_hours))
        
        # Objective fucntcion (minimise the bill i.e. electricity cost to consumers + inconvinence)
    
        Bill_optimization += lpSum((-clustered_profile[t]+shifted_load[t]) * triggers[t] for t in range(num_hours))+total_shift*inconvenience_cost
     
        # Constraint 1 Total energy reduction limitted to 5 % of total baseline consumption
  
        for d in range(num_days):
            Bill_optimization += lpSum(shifted_load[t+d*daily_slots] for t in range(daily_slots)) >= (1-0.05)*clustered_profile[d*daily_slots:(d+1)*daily_slots].sum() 
    
        Bill_optimization.solve()
    
        opt_load = pd.DataFrame(index=range(len(shifted_load)), columns=range(1))
        
    
    print("Demand optimisation Status:", LpStatus[Bill_optimization.status]) 

    for i in range(len(shifted_load)):
        opt_load.iloc[i,0]=shifted_load[i].varValue
        
    return opt_load
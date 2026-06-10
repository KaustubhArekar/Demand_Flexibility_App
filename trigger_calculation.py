def trigger_calculation(pu_gen_cost, tariff,mode,n):
    import numpy as np
    import pandas as pd

    # pricing signal calculation based on slot wise per unit generation cost (Step fucntion)
#     print(pu_gen_cost)
    gencdc = pu_gen_cost.sort_values().reset_index()
    gen_cost_dc = gencdc.iloc[:,1]
    
    mean_cost_ub = gen_cost_dc[int(len(gen_cost_dc)*0.70)] # top 0.4 % slots with higher pu_gen_cost
    mean_cost_lb = gen_cost_dc[int(len(gen_cost_dc)*0.30)] # bottom 0.5 % slots with higher pu_gen_cost   
    
#     step_size = 0.15
    triggers = pd.DataFrame(index = range(len(pu_gen_cost)), columns = tariff.columns)
    DF_trigger =np.zeros(len(pu_gen_cost)) 
    DR_incentive =np.zeros(len(pu_gen_cost)) 
    
    for s in range(len(pu_gen_cost)):
                
        if pu_gen_cost[s] >=mean_cost_ub:
            DR_incentive [s] =round((pu_gen_cost[s] - mean_cost_ub)/step_size)
            DF_trigger [s] =round((pu_gen_cost[s] -mean_cost_ub)/step_size)
        elif pu_gen_cost[s]<=mean_cost_lb:
            DF_trigger [s] = round((pu_gen_cost[s] - mean_cost_ub)/step_size)
        else:
            DR_incentive [s] = 0
            DF_trigger [s] = 0
     
    
#     plt.plot(DR_incentive*DR_lambda)
    for i in range(len(tariff.columns)):
        if mode[i]=='DR':
            triggers.iloc[:,i] =  DR_incentive[:]*(base_incenitve_DR +n* DR_lambda) + tariff.iloc[:,i]
        elif mode[i] =='DF':
            triggers.iloc[:,i] = tariff.iloc[:,i] + DF_trigger*(base_incenitve_DF+n*DF_lambda)
        else:
            triggers.iloc[:,i] = DR_incentive[:]*(base_incenitve_DR+n* DR_lambda)
                
            
    return triggers
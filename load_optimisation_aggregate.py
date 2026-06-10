def load_optimisation_aggregate(cluster_triggers,mode,flexibility,clustered_profile,tarifff,inconvenience_cost):
    import time
    import pandas as pd
    from load_optimisation import load_optimisation

    total_slots = len(tariff)  

    num_columns = len(clustered_profile.columns)


    optimized_clusters = pd.DataFrame(columns = clustered_profile.columns, index = range(len(clustered_profile)))
    load_billing= pd.DataFrame(columns = clustered_profile.columns, index = range(len(clustered_profile)))
    start = time.time()
    for c in range(num_columns):
        cluster_profile = clustered_profile.iloc[:,c].reset_index(drop=True)
        tariff_structure =cluster_triggers.iloc[:,c].reset_index(drop=True)
        opt_profile = load_optimisation(tariff_structure, cluster_profile, flex[c],mode[c],inconvenience_cost[c])
        optimized_clusters.iloc[:,c] = opt_profile[0]
        if mode[c] == 'DRR':
            load_billing.iloc[:,c] = opt_profile[0] - cluster_profile
        elif mode[c] =='DF':
            load_billing.iloc[:,c] = opt_profile[0] - cluster_profile
        else:
            load_billing.iloc[:,c] = optimized_clusters.iloc[:,c]
                 
    end = time.time()
    
    tff = cluster_triggers
    tff.columns = optimized_clusters.columns
    
 
    bills = 1000*load_billing.mul(tff)
    
    for c in range(num_columns):
        if mode[c] =='DRR':
            bills.iloc[:,c] = bills.iloc[:,c]+1000*tarifff.iloc[:,c]*optimized_clusters.iloc[:,c]
        elif mode[c] =='DF':
            bills.iloc[:,c] = bills.iloc[:,c]+1000*tarifff.iloc[:,c]*optimized_clusters.iloc[:,c]
    
    return optimized_clusters, bills
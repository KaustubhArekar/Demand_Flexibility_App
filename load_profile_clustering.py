def load_profile_clustering(tariff,path_for_meter_data):
    import os
    import random
    import time

    import numpy as np
    import pandas as pd

    start_time_cl = time.time()
    cwd = os.path.abspath(path_for_meter_data)
    files = os.listdir(cwd) 
    print(files)
    tarifff = pd.DataFrame()
    ci=0
    final_clusters = pd.DataFrame()
    for file in files:
        
        if file.endswith('.csv'):
            print(f"Processing file: {file}") 
            file_path = os.path.join(cwd, file) 
            data1 = pd.read_csv(file_path)
    
            data_load = data1.drop(columns ='slots')
#             data_load = data1
            max_load = data_load.max()
            norm_mvd = data_load/max_load
            number_of_meters = len(data_load.columns)
#             clusters_rqd = int(input("Please enter no. of clusters required for " + str(file).strip('.csv') + " category:"))
            cluster_centroids = pd.DataFrame(index=range(len(norm_mvd)),columns = range(clusters_rqd[ci]))
            cluster_tariff = pd.DataFrame(index=range(len(norm_mvd)),columns = range(clusters_rqd[ci]))
            cluster_centroids = norm_mvd.iloc[:,[random.randint(1, number_of_meters-1) for _ in range(clusters_rqd[ci])]]
            
            #meter list
            meter_list_1 = norm_mvd.columns.tolist()
            #cost matrix
            cost_matrix = pd.DataFrame(index = range(len(cluster_centroids.columns)), columns = meter_list_1)
            
            #iterative clustering
            number_of_iterations = 15 
            
            for it in range(number_of_iterations):
                for i in range(number_of_meters):
                
                    for n in range(len(cluster_centroids.columns)):
                        cost_matrix.iloc[n,i] = sum(norm_mvd.iloc[:,i] - cluster_centroids.iloc[:,n])**2
                
                    min_value_indices = np.argmin(np.array(cost_matrix), axis=0)
            
                for nn in range(len(cluster_centroids.columns)): 
                    indexes = np.where(min_value_indices == nn)[0]
                    if it == number_of_iterations-1:
                        cluster_centroids.iloc[:,nn] = data_load.iloc[:,indexes].mean(axis=1)
                    else:
                        cluster_centroids.iloc[:,nn] = norm_mvd.iloc[:,indexes].mean(axis=1)  
                      
            
            clustered_meters_list = pd.DataFrame(index = range(1),columns = ['C' + str(i) for i in range(1, clusters_rqd[ci]+1)])
            for ii in range (len(cluster_centroids.columns)):
                indexes = np.where(min_value_indices == ii)[0]
                clustered_meters_list.iloc[0,ii] = indexes  
            
            # cluster_centroids.columns = [str(file).strip('.csv') +'_Cluster-' + str(i) for i in range(1, clusters_rqd+1)]            
            cluster_centroids.columns = [(str(file))[:3] +'_Cluster-' + str(i) for i in range(1, clusters_rqd[ci]+1)]
            categorywise_cluster = cluster_centroids.to_dict()
            for i in range(clusters_rqd[ci]):
                tarifff = pd.concat([tarifff,tariff[os.path.splitext(file)[0]]],axis=1)
            final_clusters = pd.concat([final_clusters,cluster_centroids],axis=1)
            
            ci=ci+1
            
    aggregate_demand_from_clusters = final_clusters.sum(axis=1)
    
    tariff_df = pd.DataFrame(columns=final_clusters.columns, index = range(len(final_clusters)))
    tariff_df.iloc[:,:] = tarifff.iloc[:,:]
    bills = final_clusters.iloc[:,:]*tariff_df.iloc[:,:]
    
            
    end_time_cl = time.time()
            
    print('Total time elapsed: ', end_time_cl -start_time_cl)
    return final_clusters, aggregate_demand_from_clusters,bills,indexes,tarifff
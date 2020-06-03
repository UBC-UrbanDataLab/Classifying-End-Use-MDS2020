#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:27:50 2020
@author: connor
"""
import pandas as pd
import numpy as np


import data_preparation
import aggregation
import clustering


def main():
    #0) Set Constants (remember, constants are named in all caps with underscores between words)
    #################
    
    # TODO: write code to create a proper list of each day in the decided upon date-range store as DATELIST
    
    
    DATELIST = ["2020-04-01","2020-04-02"]
    
    ##################### Connor Update ##################### TEMP
    DATELIST = ["2020-03-16","2020-05-01"] # These dates are in the test_data folder so this is just here for testing purposes
    ##################### Connor Update ##################### TEMP

    SENSOR_ID_TAGS = [1,2,3,4,5,6] # order is ["groupRef","equipRef","navName","siteRef","typeRef","unit"] #NOTE: Including "unit" here means that we WILL have inconsistent units after aggregations unless we address them in the for loop BEFORE running agg_all, it's fine for now but this will need to be addressed
                                 # Contiued from above: including "unit" causes issues when there are duplicate items with mixed units (need to run the code to fix the units during this for loop or ignore units in the clustering phase)
    # The planned update to the InfluxDB may change SENSOR_ID_TAGS to only [1] as in ["uniqueID"]
    
    #1) Cluster NC data
    ###################
    
    # 1a) load+aggregate NC data (including weather), grouping by sensor ID fields [and 'unit'?]
    # DONE: Write data_preparation.query_csv()
    # TODO: Check that last_idx_to_col is supposed to be True for aggregation.agg_all()
    # TODO: Update call to aggregation.append_agg() once that function is finalized
    # TODO: Make sure col names are correct when working with nc_data df in last two lines
    last_idx_as_cols = False
    is_first_iter = True
    cnt=1
    for day in DATELIST:
        temp_df = data_preparation.query_csv(client=None, date=day, site=None)
        if temp_df is None: # Changed from temp_df==None to temp_df is None (was giving an error as is)
            continue
        temp_df = aggregation.split_datetime(temp_df) # Added to create month and hour columns (must have at least hour for aggs)
        if is_first_iter:
            struct_df = temp_df.head(1) # Added b/c need the structure of the df prior to any aggregations for append_agg
            #SENSOR_ID_TAGS.append(temp_df.columns.tolist().index("hour")) # Added b/c bool and cat agg functions need this
            nc_data = aggregation.agg_all(temp_df, how="all", col_idx=SENSOR_ID_TAGS, last_idx_to_col=last_idx_as_cols) # Rearanged this way b/c need is_first_itter condition to update SENSOR_ID_TAGS to include the hour column (needed for the categorical and boolean aggregation functions to work)
            is_first_iter = False
            #if not last_idx_to_col:
            #    nc_data = nc_data.drop('hour', axis=1)
        else:
            temp_df = aggregation.agg_all(temp_df, how="all", col_idx=SENSOR_ID_TAGS, last_idx_to_col=last_idx_as_cols) # Updated num_how to how (changed in original since how is required for all aggregation functions now)
            #if not last_idx_to_col:
                # This if statement is to get rid of hour if it isn't being grouped on (causing 0 values)
            #    temp_df = temp_df.drop('hour', axis=1)
            nc_data = aggregation.append_agg(df1=temp_df, df2=nc_data, df=struct_df, col_idx=SENSOR_ID_TAGS, last_idx_to_col=last_idx_as_cols)
        cnt += 1
    temp_df = None
    nc_data["update_rate"] = nc_data["count"] / cnt # Changed obsv_count to count b/c the pandas .agg function output defaults to count
    nc_data.drop("count", inplace=True, axis=1) # Added axis=1 b/c default axis=0 which is rows, axis=1 is for columns

    # b) Encode and scale NC data
    # TODO: Look up the correct function name for fixing units of measurement
    # TODO: 
    #clean and correct units of measurement
#    nc_data=some_call_to_fix_units_function(nc_data) # Either this needs to be done sooner, we need to join the units back on after the above grouping, or we need to make sure that this funciton does an aggregation of sorts (would need to keep the "count" column)
    # Continued from the comment on the above function: The easiest way to do this may be to not include "unit" in the aggregation and join the list of the "correct" units with the sensors after the aggregation
    cont_cols = [i for i in range(6,len(nc_data.columns))] # used in clustering but need before units are encoded
    nc_data=data_preparation.encode_units(nc_data) # Uncertain if we will need to encode units yet (try with and without and see)
    # c) cluster NC data to get df of sensor id fields + cluster group number
    gow_dist = clustering.calc_gowers(nc_data, cont_cols)
    mds_data = clustering.multidim_scale(gow_dist, num_dim=2)
    #clusters = clustering.cluster(gow_dist, 'hdbscan', continuous_columns = cont_cols, input_type='gowers')
    clusters = clustering.cluster(mds_data, 'meanshift', continuous_columns = cont_cols, input_type='mds')
    
    testClusts = pd.DataFrame(clusters, columns=["cluster"]) ################### Test
    testClusts['cluster'].value_counts().sort_index() ################### Test
    
    test_idx = [i for i in range(len(SENSOR_ID_TAGS))]
    test_groups = nc_data.columns[test_idx].values.tolist()
    test_groups.append("cluster")
    drop_cols = list(set(nc_data.columns.tolist())-set(test_groups))
    nc_data_w_clusters = pd.concat([nc_data, testClusts], axis=1)
    cluster_groups = nc_data_w_clusters
    cluster_groups = cluster_groups.drop(drop_cols, axis=1)
    
    #avg_clust_update_rate = nc_data_w_clusters.groupby(['cluster']).agg({'update_rate':'mean'},axis=1)
    
    
    
    
    # d) Reload NC data + join cluster group num + aggregate, this time grouping by date, time, and clust_group_num
    last_idx_as_cols = True
    is_first_iter = True
    cnt=1
    for day in DATELIST:
        temp_df = data_preparation.query_csv(client=None, date=day, site=None)
        if temp_df is None: # Changed from temp_df==None to temp_df is None (was giving an error as is)
            continue
        temp_df = aggregation.split_datetime(temp_df) # Added to create month and hour columns (must have at least hour for aggs)
        temp_df = temp_df.merge(cluster_groups, how='left', on=cluster_groups.columns[:-1].tolist())
        #print(temp_df.columns.tolist())
        if is_first_iter:
            update_rates = temp_df.groupby(['hour','date','cluster']).agg({'value':'count'},axis=1)
            struct_df = temp_df.head(1) # Added b/c need the structure of the df prior to any aggregations for append_agg
            CLUSTER_ID_TAGS = [temp_df.columns.tolist().index("hour"), temp_df.columns.tolist().index("date"), temp_df.columns.tolist().index("cluster")]
            #SENSOR_ID_TAGS.append(temp_df.columns.tolist().index("hour"))
            #SENSOR_ID_TAGS.append(temp_df.columns.tolist().index("date"))
            #SENSOR_ID_TAGS.append(temp_df.columns.tolist().index("cluster"))
            nc_data = aggregation.agg_all(temp_df, how="all", col_idx=CLUSTER_ID_TAGS, last_idx_to_col=last_idx_as_cols) # Rearanged this way b/c need is_first_itter condition to update SENSOR_ID_TAGS to include the hour column (needed for the categorical and boolean aggregation functions to work)
            #print(nc_data.describe())
            #print(nc_data['count'].describe())
            is_first_iter = False
            #if not last_idx_to_col:
            #    nc_data = nc_data.drop('hour', axis=1)
        else:
            update_rates = pd.concat([update_rates, temp_df.groupby(['hour','date','cluster']).agg({'value':'count'},axis=1)])
            temp_test = temp_df.copy()
            temp_df = aggregation.agg_all(temp_df, how="all", col_idx=CLUSTER_ID_TAGS, last_idx_to_col=last_idx_as_cols) # Updated num_how to how (changed in original since how is required for all aggregation functions now)
            #print(temp_df.describe())
            #print(temp_df['count'].describe())
            #if not last_idx_to_col:
                # This if statement is to get rid of hour if it isn't being grouped on (causing 0 values)
            #    temp_df = temp_df.drop('hour', axis=1)
            nc_data = aggregation.append_agg(df1=temp_df, df2=nc_data, df=struct_df, col_idx=CLUSTER_ID_TAGS, last_idx_to_col=last_idx_as_cols)
        cnt += 1
        
    test_update_rates = update_rates.unstack()
    test_update_rates.columns = test_update_rates.columns.droplevel(level=0)
    test_update_rates = test_update_rates.fillna(0)
    
    sensor_count_per_cluster = cluster_groups.groupby('cluster').agg({cluster_groups.columns.tolist()[0]:'count'})
    sensor_count_per_cluster.columns = ['count']
    
    for cluster in cluster_groups['cluster'].unique():
        test_update_rates.loc[:,cluster] = test_update_rates.loc[:,cluster]/sensor_count_per_cluster.loc[cluster][0]
    
    test_update_rates = test_update_rates.add_prefix('urate_')
    
    nc_data = nc_data.join(test_update_rates, how='left', on=['hour', 'date'])
    
    nc_data.to_csv('sample_cluster_output.csv')
        
    #2) Model EC/NC relationship
    ############################
    #    temp) Make a fake version of the output dataframe from step 1 so that step 2 can be (mostly) developed 
    #        without waiting for step 1 to be finished!
        
    #    a) Load+aggregate EC data, grouping by date, time, and sensor ID fields (no feature selection needed yet!)
    #    b) Also create second DF by aggregating further just using sensor ID fields (end result=1row per sensor)
    #    c) For each unique EC sensorID (i.e. row in 2b_EC_data_df), create LASSO model using 2a_EC_data_df and 
    #       step1_output_NC_data_df. Model is basically: Y=EC response and Xn=NC data
    
    #    d) Join the coeffecients from LASSO model to 2b_EC_data_df (so each EC sensor has a list of n coeffecients)
    
    #    OUTPUT OF STEP2 = dataframe with EC sensor ID fields, mean response, and all n coeffecients from 
    #        that unique EC sensor's LASSO model
"""
    3) Mid-Process cleanup
    ######################
        a) set all NC data dataframes = None (clear up memory)
        b) set 2a_EC_data_df = None (clear up memory)
        [c) we could also save any EC dataframes to temporary csvs if we want to split the program up and allow a user 
            to just run only the first two steps which maybe take a long time if all data has to be queried? That would
            make this step a checkpoint of sorts...just a thought]
        OUTPUT OF STEP = nothing! Just more available memory.
    
    4) Prep EC data for classification model 
    ########################################
        a) Load metadata and join with 2b_EC_data_df
        b) Apply feature selection function(s) to the joined EC+metadata
        c) Encode and scale the EC+metadata
        d) Join the model coeffecients from step2 output to the EC+metadata
        OUTPUT OF STEP = dataframe with EC sensor ID fields, selected EC features, model coeffecients
    
    5) Classification model 
    #######################
        a) Run classification model on output from step 4
        b) ?
        c) PROFIT!
        OUTPUT OF STEP = dataframe with EC sensor ID fields and end-use group
    """


################ Used for calibrating dbscan and hdbscan clustering
from matplotlib import pyplot as plt
sorted_vals = gow_dist[0]
sorted_vals = sorted_vals[np.argsort(-gow_dist[0])]
plt.plot(sorted_vals)

testGow = pd.DataFrame(sorted_vals, columns=["values"]) ################### Test
testGow['values'].value_counts()#.sort_index() ################### Test

#################### Visualizing the clusters
# Make Data into a Dataframe
data_2d_df = pd.DataFrame(data=mds_data, columns = ['x','y'])
data_2d_df['cluster'] = testClusts['cluster']

# Plotting the dbscan clusters (Fit pre-MDS)
plt.scatter(data_2d_df['x']*1000,data_2d_df['y']*1000, c=data_2d_df['cluster'])

hdb_2d = data_2d_df.copy()
meanshift_2d = data_2d_df.copy()
data_2d_df_w_units = data_2d_df.copy()

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
    
    DATELIST = ["2020-03-16","2020-05-01"] # These dates are in the test_data folder so this is just here for testing purposes

    SENSOR_ID_TAGS = [1,2,3,4,5,6] # order is ["groupRef","equipRef","navName","siteRef","typeRef","unit"] 
    #NOTE: Including "unit" here means that we WILL have inconsistent units after aggregations unless we address
    # them in the for loop BEFORE running agg_all, it's fine for now but this will need to be addressed
    # Including "unit" causes issues when there are duplicate items with mixed units (need to run the code to fix the
    # units during this for loop or ignore units in the clustering phase)

    # The planned update to the InfluxDB may change SENSOR_ID_TAGS to only [1] as in ["uniqueID"]
    
    #1) Cluster NC data
    ###################
    
    # a) load+aggregate NC data (including weather), grouping by sensor ID fields [and 'unit'?]

    # DONE: Write data_preparation.query_csv()
    # DONE: Check that last_idx_to_col is supposed to be True for aggregation.agg_all() (False for pre-cluster aggregation [if clustering on overall measurements], and True for the post cluster aggregation)
    # TODO: Determine what aggregation period to cluster for (Current is just overall values, could hour of the day, 6 hour increments, 12 hour increments, etc...) (I think every hour will likely be too computationally expensive if we need to calculate Gower's distance, and will be if we need MDS)
    # DONE: Update call to aggregation.append_agg() once that function is finalized
    # DONE: Make sure col names are correct when working with nc_data df in last two lines
    last_idx_as_cols = False
    is_first_iter = True
    cnt=1
    for day in DATELIST:
        # Querying and preping data for aggregations
        temp_df = data_preparation.query_csv(client=None, date=day, site=None)
        if temp_df is None:
            continue
        temp_df = aggregation.split_datetime(temp_df)
        if is_first_iter:
            # Creating a low memory dataframe for the append_agg function before the structure is changed by agg_all
            struct_df = temp_df.head(1)
            # Aggregating the first date's data
            nc_data = aggregation.agg_all(temp_df, how="all", col_idx=SENSOR_ID_TAGS, last_idx_to_col=last_idx_as_cols)
            is_first_iter = False
        else:
            # Aggregating the current date's data and aggregate it with the current running total
            temp_df = aggregation.agg_all(temp_df, how="all", col_idx=SENSOR_ID_TAGS, last_idx_to_col=last_idx_as_cols)
            nc_data = aggregation.append_agg(df1=temp_df, df2=nc_data, struct_df=struct_df, col_idx=SENSOR_ID_TAGS, last_idx_to_col=last_idx_as_cols)
        cnt += 1
    # Freeing up some memory
    temp_df = None
    # Calculating the update rate
    nc_data["update_rate"] = nc_data["count"] / cnt
    nc_data.drop("count", inplace=True, axis=1)

    
    # b) Encode and scale NC data
    # TODO: Look up the correct function name for fixing units of measurement (getting added to the query function, can remove/update to DONE once confirmed complete)
    # TODO: clean and correct units of measurement (getting added to the query function, can remove/update to DONE once confirmed complete)
    # DONE: Encode categorical variables
    # TODO: Test performance for categorical variables included vs excluded
    # TODO: Scale continuous variables
    # TODO: Test performance for continuous variables scaled vs not scaled
    
    # Getting Indexes of the continuous columns
    cont_cols = [i for i in range(len(SENSOR_ID_TAGS),len(nc_data.columns))]
    
    # Scale Continuous
    scaled_data = data_preparation.scale_continuous(nc_data, cont_cols)
    nc_data = pd.concat([nc_data.iloc[:, 0:len(SENSOR_ID_TAGS)], pd.DataFrame(scaled_data, index=nc_data.index, columns=nc_data.columns[cont_cols].tolist())], axis=1)
    
    # Encoding units
    nc_data = data_preparation.encode_units(nc_data)
    
    # c) cluster NC data to get df of sensor id fields + cluster group number
    # DONE: Calculate Gower's distance
    # TODO: Determine if the model we are using requires Gower's distance (if not including categorical variables then may not need it)
    # DONE: Calculate MDS (if not including categorical variables then may not need it)
    # TODO: Determine if the model we are using requires MDS
    # DONE: Cluster data
    # TODO: Determine which clustering model to use
    # DONE: Create dataframe that relates the unique sensors to the relevant cluster
    
    # Calculating Gower's Distance, MDS, and clustering
    gow_dist = clustering.calc_gowers(nc_data, cont_cols)
    mds_data = clustering.multidim_scale(gow_dist, num_dim=2)
    #clusters = clustering.cluster(gow_dist, 'hdbscan', continuous_columns = cont_cols, input_type='gowers')
    clusters = clustering.cluster(mds_data, 'meanshift', continuous_columns = cont_cols, input_type='mds')
    
    ##################################################################################
    ###################################### TEST ######################################
    
    # Just for testing purposes (shows how many observations belong to each cluster)
    cluster_df = pd.DataFrame(clusters, columns=["cluster"]) #################### Test
    cluster_df['cluster'].value_counts().sort_index() ########################### Test
    
    ###################################### TEST ######################################
    ##################################################################################
    
    # Generating a list of the columns to keep when making the dataframe relating sensors to clusters(the unique identifiers for an NC sensor and cluster)
    unique_cols_idx = [i for i in range(len(SENSOR_ID_TAGS))]
    unique_cols = nc_data.columns[unique_cols_idx].values.tolist()
    unique_cols.append("cluster")
    # Creating dataframe that identifies which unique sensors belong to which cluster
    drop_cols = list(set(nc_data.columns.tolist())-set(unique_cols))
    cluster_groups = pd.concat([nc_data, pd.DataFrame(clusters, columns=["cluster"])], axis=1)
    cluster_groups = cluster_groups.drop(drop_cols, axis=1)
    
    # d) Reload NC data + join cluster group num + aggregate, this time grouping by date, time, and clust_group_num
    # DONE: Load NC data and join the cluster groups to it
    # DONE: Aggregate the NC data by hour, date, and cluster group but with cluster groups as columns
    # DONE: Calculate average instrument update rate per cluster per hour per day
    last_idx_as_cols = True
    is_first_iter = True
    cnt=1
    for day in DATELIST:
        # Querying and preping data for aggregations
        temp_df = data_preparation.query_csv(client=None, date=day, site=None)
        if temp_df is None:
            continue
        temp_df = aggregation.split_datetime(temp_df) # Added to create month and hour columns (must have at least hour for aggs)
        temp_df = temp_df.merge(cluster_groups, how='left', on=cluster_groups.columns[:-1].tolist())
        if is_first_iter:
            # Calculating the count of sensor updates per hour per day per cluster for the first date
            update_rates = temp_df.groupby(['hour','date','cluster']).agg({'value':'count'},axis=1)
            # Creating a low memory dataframe for the append_agg function before the structure is changed by agg_all (different structure required than previous)
            struct_df = temp_df.head(1)
            # Identifying the indexes of the items being aggregated on (hour, date, and cluster)
            CLUSTER_ID_TAGS = [temp_df.columns.tolist().index("hour"), temp_df.columns.tolist().index("date"), temp_df.columns.tolist().index("cluster")]
            # Aggregating the first date's data
            nc_data = aggregation.agg_all(temp_df, how="all", col_idx=CLUSTER_ID_TAGS, last_idx_to_col=last_idx_as_cols)
            is_first_iter = False
        else:
            # Calculating the count of sensor updates per hour per day per cluster for the current date
            update_rates = pd.concat([update_rates, temp_df.groupby(['hour','date','cluster']).agg({'value':'count'},axis=1)])
            # Aggregating the current date's data and aggregate it with the current running total
            temp_df = aggregation.agg_all(temp_df, how="all", col_idx=CLUSTER_ID_TAGS, last_idx_to_col=last_idx_as_cols)
            nc_data = aggregation.append_agg(df1=temp_df, df2=nc_data, struct_df=struct_df, col_idx=CLUSTER_ID_TAGS, last_idx_to_col=last_idx_as_cols)
        cnt += 1
    
    # Re-format update rates so that clusters are columns
    update_rates = update_rates.unstack()
    update_rates.columns = update_rates.columns.droplevel(level=0)
    update_rates = update_rates.fillna(0)
    
    # Calculate the number of sensors per cluster
    sensor_count_per_cluster = cluster_groups.groupby('cluster').agg({cluster_groups.columns.tolist()[0]:'count'})
    sensor_count_per_cluster.columns = ['count']
    
    # Calculate the average sensor update rate per hour per cluster
    for cluster in cluster_groups['cluster'].unique():
        update_rates.loc[:,cluster] = update_rates.loc[:,cluster]/sensor_count_per_cluster.loc[cluster][0]
    
    # Rename the update rate columns and join them to the nc_data
    update_rates = update_rates.add_prefix('urate_')
    nc_data = nc_data.join(update_rates, how='left', on=['hour', 'date'])
    nc_data = nc_data.drop('count', axis=1)
    
    ##################################################################################
    ###################################### TEMP ######################################
    # Just here for writing sample data to csv for testing in other sections
    nc_data.to_csv('sample_cluster_output.csv')
    ###################################### TEMP ######################################
    ##################################################################################
    
    
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
data_2d_df['cluster'] = cluster_groups['cluster']

# Plotting the dbscan clusters (Fit pre-MDS)
plt.scatter(data_2d_df['x']*1000,data_2d_df['y']*1000, c=data_2d_df['cluster'])

hdb_2d = data_2d_df.copy()
meanshift_2d = data_2d_df.copy()
data_2d_df_w_units = data_2d_df.copy()

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


def main():
    #0) Set Constants (remember, constants are named in all caps with underscores between words)
    #################
    
    # TODO: write code to create a proper list of each day in the decided upon date-range store as DATELIST
    
    
    DATELIST = ["2020-04-01","2020-04-02"]

    SENSOR_ID_TAGS = [1,2,3,4,5] #I think this is ["groupRef","equipRef","navName","typeRef","unit"]
    # The planned update to the InfluxDB may change SENSOR_ID_TAGS to only [1] as in ["uniqueID"]
    
    #1) Cluster NC data
    ###################
    
    # 1a) load+aggregate NC data (including weather), grouping by sensor ID fields [and 'unit'?]
    # TODONE: Write data_preparation.query_csv()
    # TODO: Check that last_idx_to_col is supposed to be True for aggregation.agg_all()
    # TODO: Update call to aggregation.append_agg() once that function is finalized
    # TODO: Make sure col names are correct when working with nc_data df in last two lines

    is_first_iter = True
    cnt=1
    for day in DATELIST:
        temp_df = data_preparation.query_csv(client=None, date=day, site=None)
        if temp_df == None:
            continue
        temp_df = aggregation.agg_all(temp_df, num_how="all", col_idx=SENSOR_ID_TAGS, last_idx_to_col=True)
        if is_first_iter: 
            nc_data=temp_df
            is_first_iter = False
        nc_data = aggregation.append_agg(newdf=temp_df, masterdf=nc_data)
        cnt += 1
    temp_df = None
    nc_data["update_rate"] = nc_data["obsv_count"] / cnt
    nc_data.drop("obsv_count", inplace=True)

    # b) Encode and scale NC data
    # TODO: Look up the correct function name for fixing units of measurement
    # TODO: 
    #clean and correct units of measurement
    #nc_data=some_call_to_fix_units_function(nc_data)
    
    #nc_data=data_preparation.encode_units()
    # c) cluster NC data to get df of sensor id fields + cluster group number
    # d) Reload NC data + join cluster group num + aggregate, this time grouping by date, time, and clust_group_num
   
        
    #2) Model EC/NC relationship
    ############################
        
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
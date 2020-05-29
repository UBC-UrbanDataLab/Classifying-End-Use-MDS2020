#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:27:50 2020

@author: connor
"""

def main():
    """
    Highlevel overview of steps:

    1) Cluster NC data
        a) load+aggregate NC data (including weather), grouping by sensor ID fields [and 'unit'?]
        b) Encode and scale NC data
        c) cluster NC data to get df of sensor id fields + cluster group number
        d) Reload NC data + join cluster group num + aggregate, this time grouping by date, time, and clust_group_num
        OUTPUT OF STEP = dataframe with clust_group_num, 5nums per agg period. 
        
    2) Model EC/NC relationship
        a) Load+aggregate EC data, grouping by date, time, and sensor ID fields (no feature selection needed yet!)
        b) Also create second DF by aggregating further just using sensor ID fields (end result=1row per sensor)
        c) For each unique EC sensorID (i.e. row in 2b_EC_data_df), create LASSO model using 2a_EC_data_df and 
            step1_output_NC_data_df. Model is basically: Y=EC response and Xn=NC data
        d) Join the coeffecients from LASSO model to 2b_EC_data_df (so each EC sensor has a list of n coeffecients)
        OUTPUT OF STEP = dataframe with EC sensor ID fields, mean response, and all n coeffecients from 
            that unique EC sensor's LASSO model

    3) Mid-Process cleanup
        a) set all NC data dataframes = None (clear up memory)
        b) set 2a_EC_data_df = None (clear up memory)
        [c) we could also save any EC dataframes to temporary csvs if we want to split the program up and allow a user 
            to just run only the first two steps which maybe take a long time if all data has to be queried? That would
            make this step a checkpoint of sorts...just a thought]
        OUTPUT OF STEP = nothing! Just more available memory.
    
    4) Prep EC data for classification model 
        a) Load metadata and join with 2b_EC_data_df
        b) Apply feature selection function(s) to the joined EC+metadata
        c) Encode and scale the EC+metadata
        d) Join the model coeffecients from step2 output to the EC+metadata
        OUTPUT OF STEP = dataframe with EC sensor ID fields, selected EC features, model coeffecients
    
    5) Classification model 
        a) Run classification model on output from step 4
        b) ?
        c) PROFIT!
        OUTPUT OF STEP = dataframe with EC sensor ID fields and end-use group
    """


### OLD FRAMEWORK...Will need update to match concept listed above ###
    ###
    # Query EC data
    
    ###
    # Query NC data
    
    ###
    # Query weather data (append to NC data)
    
    ###
    # Read in SkySpark metadata
    #  
    
    ###
    # Join EC data with SkySpark metadata
    #

    ###
    # Join NC data with SkySpark metadata
    # 

    ###
    # Aggregate "values" and calculate sensor update rate
    
    # Identify any groupings in the sensor tags/names
    
    # Categorical Feature selection
    
    # Continuous Feature selection
    
    # Drop unwanted categorical and continuous features
    
    # Scale "values" and encode categorical features
    
    #################################
    # Cluster NC sensors
    
    # Join NC and EC Datasets
    
    #################################
    
    # Seperate labeled/unlabeled data
    
    # Train model/load (do we need to train the model every time, could we just save and load it?)
    
    # Predict NRCan Labels
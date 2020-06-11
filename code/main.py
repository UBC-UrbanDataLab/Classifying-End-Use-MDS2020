#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:27:50 2020

@author: connor
"""
### ~ Library Imports ~ ###
# Data Formatting and Manipulation Imports
import pandas as pd
import numpy as np

# Supervised Learning Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Project Module Imports
import data_preparation
import aggregation
import clustering


def main():
    #0) Set Constants (remember, constants are named in all caps with underscores between words)
    #################

    # TODO: write code to create a proper list of each day in the decided upon date-range store as DATELIST

    ##############################################################################################################
    ############### TEMP: for testing from csvs to delete once actual querying code is implemented ###############
    ###############       (gets list of file names in the given path)
    from os import listdir
    from os.path import isfile, join
    mypath = 'test_data'
    DATELIST = [f.split(".")[0] for f in listdir(mypath) if isfile(join(mypath, f))]
    #DATELIST = ["2020-03-16","2020-05-01"] # These dates are in the test_data folder so this is just here for testing purposes
    ############### TEMP: for testing from csvs to delete once actual querying code is implemented ###############
    ##############################################################################################################

    SENSOR_ID_TAGS = [1,2,3,4,5,6] # order is ["groupRef","equipRef","navName","siteRef","typeRef","unit"] #NOTE: Including "unit" here means that we WILL have inconsistent units after aggregations unless we address them in the for loop BEFORE running agg_all, it's fine for now but this will need to be addressed
                                 # Contiued from above: including "unit" causes issues when there are duplicate items with mixed units (need to run the code to fix the units during this for loop or ignore units in the clustering phase)

    # The planned update to the InfluxDB may change SENSOR_ID_TAGS to only [1] as in ["uniqueID"]

    #1) Cluster NC data
    ###################
    print("####### ~~~~~ Started - Step 1: Clustering Phase ~~~~~ #######") ############### TEMP: For Tracking test progress
    # a) load+aggregate NC data (including weather), grouping by sensor ID fields [and 'unit'?]
    # TODO: Determine what aggregation period to cluster for (Current is just overall values, could hour of the day, 6 hour increments, 12 hour increments, etc...) (I think every hour will likely be too computationally expensive if we need to calculate Gower's distance, and will be if we need MDS)
    # TODO: Update to include weather data (should be as simple as expanding the query to include weather data, or having a seperate query for weather data and appending the weather dataframe to the bottom)
    print("\t##### ~~~ Started - Aggregation Phase 1 ~~~ #####") ############### TEMP: For Tracking test progress
    last_idx_as_cols = False
    is_first_iter = True
    cnt=1
    for day in DATELIST:
        print("\t\t"+str(cnt)+": "+str(day)) ############### TEMP: For Tracking test progress
        # Querying and preping data for aggregations
        temp_df = data_preparation.query_csv(client=None, date=day, site=None)
        col_names = ['datetime']
        col_names.extend(temp_df.columns[1:])
        temp_df.columns = col_names
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
        if cnt == 15: ############### TEMP: For speeding up testing of updated code for main function delete once updates confirmed to work
            break ############### TEMP: For speeding up testing of updated code for main function delete once updates confirmed to work

    print("\t\t### ~ Started - Agg Phase 1: Calculating Update Rates ~ ###") ############### TEMP: For Tracking test progress
    # Freeing up some memory
    temp_df = None
    # Calculating the update rate
    nc_data["update_rate"] = nc_data["count"] / cnt
    nc_data.drop("count", inplace=True, axis=1)


    #nc_data.to_csv('aggregated_no_units_data.csv') ############### TEMP: Write aggregation output to file to provide sample data for testing step 2, remove for final model


    # b) Encode and scale NC data
    # TODO: Look up the correct function name for fixing units of measurement (getting added to the query function, can remove/update to DONE once confirmed complete)
    # TODO: clean and correct units of measurement (getting added to the query function, can remove/update to DONE once confirmed complete)
    # TODO: Test performance for categorical variables included vs excluded
    # TODO: Scale continuous variables
    # TODO: Test performance for continuous variables scaled vs not scaled

    print("\t##### ~~~ Started - Clustering Phase ~~~ #####") ############### TEMP: For Tracking test progress

    # Getting Indexes of the continuous columns
    cont_cols = [i for i in range(len(SENSOR_ID_TAGS),len(nc_data.columns))]

    # Scale Continuous
    scaled_data = data_preparation.scale_continuous(nc_data, cont_cols)
    nc_data = pd.concat([nc_data.iloc[:, 0:len(SENSOR_ID_TAGS)], pd.DataFrame(scaled_data, index=nc_data.index, columns=nc_data.columns[cont_cols].tolist())], axis=1)

    # Encoding units
    nc_data = data_preparation.encode_units(nc_data)

    # c) cluster NC data to get df of sensor id fields + cluster group number
    # TODO: Determine if the model we are using requires Gower's distance (if not including categorical variables then may not need it)
    # TODO: Determine if the model we are using requires MDS
    # TODO: Determine which clustering model to use

    # Calculating Gower's Distance, MDS, and clustering
    gow_dist = clustering.calc_gowers(nc_data, cont_cols)
    #pd.DataFrame(gow_dist).to_csv('gowers_scaled_no_units_data.csv') ############### TEMP: Write aggregation output to file to provide sample data for testing step 2, remove for final model
    mds_data = clustering.multidim_scale(gow_dist, num_dim=2)
    #pd.DataFrame(mds_data).to_csv('mds_2d_scaled_no_units_data.csv') ############### TEMP: Write aggregation output to file to provide sample data for testing step 2, remove for final model
    clusters = clustering.cluster(mds_data, 'vbgm', num_clusts=35, input_type='mds')

    ###############################################################################
    ############### TEMP: Included for testing the clustering model ###############
    ###############       Shows how many observations belong to each cluster
    ###############       Can delete once done testing clustering model
    #cluster_df = pd.DataFrame(clusters, columns=["cluster"])
    #cluster_df['cluster'].value_counts().sort_index()
    ############### TEMP: Included for testing the clustering model ###############
    ###############################################################################

    # Generating a list of the columns to keep when making the dataframe relating sensors to clusters(the unique identifiers for an NC sensor and cluster)
    unique_cols_idx = [i for i in range(len(SENSOR_ID_TAGS))]
    unique_cols = nc_data.columns[unique_cols_idx].values.tolist()
    unique_cols.append("cluster")
    # Creating dataframe that identifies which unique sensors belong to which cluster
    drop_cols = list(set(nc_data.columns.tolist())-set(unique_cols))
    cluster_groups = pd.concat([nc_data, pd.DataFrame(clusters, columns=["cluster"])], axis=1)
    cluster_groups = cluster_groups.drop(drop_cols, axis=1)

    print("\t##### ~~~ Started - Aggregation Phase 2 ~~~ #####") ############### TEMP: For Tracking test progress

    # d) Reload NC data + join cluster group num + aggregate, this time grouping by date, time, and clust_group_num
    last_idx_as_cols = True
    is_first_iter = True
    cnt=1
    for day in DATELIST:
        print("\t\t"+str(cnt)+": "+str(day)) ############### TEMP: For Tracking test progress
        # Querying and preping data for aggregations
        temp_df = data_preparation.query_csv(client=None, date=day, site=None)
        col_names = ['datetime']
        col_names.extend(temp_df.columns[1:])
        temp_df.columns = col_names
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
            cluster_id_tags = [temp_df.columns.tolist().index("hour"), temp_df.columns.tolist().index("date"), temp_df.columns.tolist().index("cluster")]
            # Aggregating the first date's data (Aggregations must be seperate b/c can't data gets too big during calculations if not)
            temp_df_aggs = aggregation.agg_all(temp_df, how="mean", col_idx=cluster_id_tags, last_idx_to_col=last_idx_as_cols)
            if 'count_x' in temp_df_aggs.columns.tolist():
                # Each aggregation type outputs a count, ensuring joins only result in 1 count (applies for all similar if statements)
                temp_df_aggs = temp_df_aggs.drop('count_y', axis=1)
                temp_df_aggs = temp_df_aggs.rename(columns={'count_x':'count'})
            temp_df_aggs = temp_df_aggs.merge(aggregation.agg_all(temp_df, how="std", col_idx=cluster_id_tags, last_idx_to_col=last_idx_as_cols), how='left', on=temp_df_aggs.columns.tolist()[:2])
            if 'count_x' in temp_df_aggs.columns.tolist():
                temp_df_aggs = temp_df_aggs.drop('count_y', axis=1)
                temp_df_aggs = temp_df_aggs.rename(columns={'count_x':'count'})
            temp_df_aggs = temp_df_aggs.merge(aggregation.agg_all(temp_df, how="max", col_idx=cluster_id_tags, last_idx_to_col=last_idx_as_cols), how='left', on=temp_df_aggs.columns.tolist()[:2])
            if 'count_x' in temp_df_aggs.columns.tolist():
                temp_df_aggs = temp_df_aggs.drop('count_y', axis=1)
                temp_df_aggs = temp_df_aggs.rename(columns={'count_x':'count'})
            nc_data = temp_df_aggs.merge(aggregation.agg_all(temp_df, how="min", col_idx=cluster_id_tags, last_idx_to_col=last_idx_as_cols), how='left', on=temp_df_aggs.columns.tolist()[:2])
            if 'count_x' in nc_data.columns.tolist():
                nc_data = nc_data.drop('count_y', axis=1)
                nc_data = nc_data.rename(columns={'count_x':'count'})
            is_first_iter = False
        else:
            # Calculating the count of sensor updates per hour per day per cluster for the current date
            update_rates = pd.concat([update_rates, temp_df.groupby(['hour','date','cluster']).agg({'value':'count'},axis=1)])
            # Aggregating the current date's data and aggregate it with the current running total
            temp_df_aggs = aggregation.agg_all(temp_df, how="mean", col_idx=cluster_id_tags, last_idx_to_col=last_idx_as_cols)
            if 'count_x' in temp_df_aggs.columns.tolist():
                temp_df_aggs = temp_df_aggs.drop('count_y', axis=1)
                temp_df_aggs = temp_df_aggs.rename(columns={'count_x':'count'})
            temp_df_aggs = temp_df_aggs.merge(aggregation.agg_all(temp_df, how="std", col_idx=cluster_id_tags, last_idx_to_col=last_idx_as_cols), how='left', on=temp_df_aggs.columns.tolist()[:2])
            if 'count_x' in temp_df_aggs.columns.tolist():
                temp_df_aggs = temp_df_aggs.drop('count_y', axis=1)
                temp_df_aggs = temp_df_aggs.rename(columns={'count_x':'count'})
            temp_df_aggs = temp_df_aggs.merge(aggregation.agg_all(temp_df, how="max", col_idx=cluster_id_tags, last_idx_to_col=last_idx_as_cols), how='left', on=temp_df_aggs.columns.tolist()[:2])
            if 'count_x' in temp_df_aggs.columns.tolist():
                temp_df_aggs = temp_df_aggs.drop('count_y', axis=1)
                temp_df_aggs = temp_df_aggs.rename(columns={'count_x':'count'})
            temp_df_aggs = temp_df_aggs.merge(aggregation.agg_all(temp_df, how="min", col_idx=cluster_id_tags, last_idx_to_col=last_idx_as_cols), how='left', on=temp_df_aggs.columns.tolist()[:2])
            if 'count_x' in temp_df_aggs.columns.tolist():
                temp_df_aggs = temp_df_aggs.drop('count_y', axis=1)
                temp_df_aggs = temp_df_aggs.rename(columns={'count_x':'count'})
            nc_data = aggregation.append_agg(df1=temp_df_aggs, df2=nc_data, struct_df=struct_df, col_idx=cluster_id_tags, last_idx_to_col=last_idx_as_cols)
        cnt += 1
        if cnt == 15: ############### TEMP: For speeding up testing of updated code for main function delete once updates confirmed to work
            break ############### TEMP: For speeding up testing of updated code for main function delete once updates confirmed to work

    print("\t\t### ~ Started - Agg Phase 2: Calculating Update Rates ~ ###") ############### TEMP: For Tracking test progress
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

    print("####### ~~~~~ Complete - Step 1: NC Aggregation and Clustering Phase ~~~~~ #######") ############### TEMP: For Tracking test progress

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
"""
    # 5) Classification model
    #######################
    #    a) Dataprep to get the step 4 data into an appropriate format for prediction
    # TODO: Ensure that the step 4 output data is in the same format as the training_data dataframe (with the NRCan labels column as null for now)

    #    b)
    # TODO: Update read_csv() path to the correct location and file name for final product
    # TODO: Remove the train_test_split() code for final product
    # TODO: Ensure that the training_data dataframe is formated the same as the step 4 output (same as TODO for part a b/c it is that important)
    training_data = pd.read_csv('../../step4_data.csv') # Reading in the training data file
    # Manipulating dataset to be in the appropriate format for creating seperate predictor and response datasets
    cols = training_data.columns.tolist()
    cols.remove('NRCan')
    cols.append('NRCan')
    training_data = training_data[cols]
    training_data = training_data.iloc[:,2:]
    training_data = training_data.fillna(0)
    # Extracting just the number from the label
    training_data['NRCan'] = training_data['NRCan'].apply(lambda x: int(x[0]))

    # Creating seperate predictor variable and response variable numpy arrays
    x = training_data.iloc[:, :-1].values
    y = training_data.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0) ############### TEMP: For testing the model, remove for final product

    #    b) Create and train the selected model
    # Creating and fitting the classifier
    # TODO: Test various models and select a final model
    # TODO: Update model to the selected model (if not RandomForrestClassifier)
    # TODO: Change x_train and y_train to x and y once final model has been selcted
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'gini') # can include random_state=0 if we want to set the seed to be constant
    classifier.fit(x_train, y_train) # Can change x_train and y_train to just x and y for final model

    #    c) Predict the outputs for the new data
    # Predicting the outputs
    # TODO: Change x_test to whatever the numpy data to be predicted is (NOTE: Must have same format as the training_data dataframe)
    y_pred = classifier.predict(x_test) # TODO: Change x_test to whatever the actual data to be predicted's variable for final model

    ########################################################################################
    ############### TEMP: Used to test the effectiveness of the chosen model ###############
    ###############       To delete for the final model, just here so we can see some actual
    ###############       output from step 5
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, log_loss
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    logloss = log_loss(y_true=y_test, y_pred=classifier.predict_proba(x_test))
    print("accuracy: "+str(accuracy))
    print("precision: "+str(precision))
    print("recall: "+str(recall))
    print("f1: "+str(f1))
    print("logloss: "+str(logloss))
    ############### TEMP: Used to test the effectiveness of the chosen model ###############
    ########################################################################################

    #    d) Join the predictions onto the  step 4 output dataframe
    # TODO: Join the predictions onto the step 4 output dataframe (or populate the NRCan column with these values)

    #    OUTPUT OF STEP = dataframe with EC sensor ID fields and end-use group


########################################################################################
############### TEMP: Used for calibrating dbscan and hdbscan clustering ###############
###############       Can delete once final model selection and calibration complete
###############                                     OR
###############       once dbscan AND hdbscan dismissed as possible models
from matplotlib import pyplot as plt
sorted_vals = gow_dist[0]
sorted_vals = sorted_vals[np.argsort(-gow_dist[0])]
plt.plot(sorted_vals)

testGow = pd.DataFrame(sorted_vals, columns=["values"])
testGow['values'].value_counts()#.sort_index()
############### TEMP: Used for calibrating dbscan and hdbscan clustering ###############
########################################################################################

##################################################################
############### TEMP: For visualizing the clusters ###############
###############       Can delete after moved to a seperate file
###############                             OR
###############       once desired plots have been obtained for
###############       the final report
from matplotlib import pyplot as plt
# Make Data into a Dataframe
data_2d_df = pd.DataFrame(data=mds_data, columns = ['x','y'])
data_2d_df['cluster'] = cluster_groups['cluster']

# Plotting the dbscan clusters (Fit pre-MDS)
plt.scatter(data_2d_df['x']*1000,data_2d_df['y']*1000, c=data_2d_df['cluster'])
############### TEMP: For visualizing the clusters ###############
##################################################################

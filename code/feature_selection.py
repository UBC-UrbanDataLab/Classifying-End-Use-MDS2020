#!/usr/bin/env python
# coding: utf-8

# In[73]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### ~ Library Imports ~ ###
# General Imports
from datetime import datetime, date, timedelta
# Data Formatting and Manipulation Imports
import pandas as pd
import numpy as np
# Feature Selection Imports
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV, SelectKBest, f_classif, mutual_info_classif
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
# Clustering Step Imports
from sklearn.cluster import AgglomerativeClustering
# Regression Step Imports
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
# Supervised Classification Step Imports
from sklearn.ensemble import BaggingClassifier
# Project Module Imports
import data_preparation
import aggregation
import clustering
    


def feature_selection():
    #0) Set Constants (remember, constants are named in all caps with underscores between words)
    display_prediction_metrics = True # Set True to display prediciton metrics (confusion matrix, accuracy, precission, recall, f1 score, logloss), else set False
    # Getting a list of the last 90 dates
    DATELIST =  [(date.today() + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(91)] # NOTE: To update with the number of days desired to pull data for (currently has 91)
    DATELIST.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d"))

    ##############################################################################################################
    ############### TEMP: for testing from csvs to delete once actual querying code is implemented ###############
    ###############       (gets list of file names in the given path)
    from os import listdir
    from os.path import isfile, join
    mypath = 'test_data'
    DATELIST = [f.split(".")[0] for f in listdir(mypath) if isfile(join(mypath, f))]
    DATELIST = DATELIST[1:3]
    ############### TEMP: for testing from csvs to delete once actual querying code is implemented ###############
    ##############################################################################################################

    SENSOR_ID_TAGS = [1,2,3,4,5,6] # order is ["groupRef","equipRef","navName","siteRef","typeRef","unit"]

    # The planned update to the InfluxDB may change SENSOR_ID_TAGS to only [1] as in ["uniqueID"]

    #1) Cluster NC data
    ###################
    print("####### ~~~~~ Started - Step 1: Clustering Phase ~~~~~ #######") ############### TEMP: For Tracking test progress
    # a) load+aggregate NC data (including weather), grouping by sensor ID fields [and 'unit'?]
    print("\t##### ~~~ Started - Step 1 a): Aggregation Phase 1 ~~~ #####") ############### TEMP: For Tracking test progress
    last_idx_as_cols = False
    is_first_iter = True
    cnt=1
    for day in DATELIST:
        print("\t\t"+str(cnt)+": "+str(day)) ############### TEMP: For Tracking test progress
        # Querying and preping data for aggregations
        temp_df = data_preparation.query_csv(client=None, date=day, site=None) ############### TEMP: To be replaced by actual query functions in final product
        weather_df = data_preparation.query_weather_csv(client=None, date=day, site=None) ############### TEMP: To be replaced by actual query functions in final product
        if weather_df is None: ############### TEMP: To be replaced by actual query functions in final product
            pass
        else:
            temp_df = pd.concat([temp_df, weather_df]) ############### TEMP: To be replaced by actual query functions in final product
            temp_df = temp_df.fillna('empty') ############### TEMP: To be replaced by actual query functions in final product # Aggregation doesn't work with nan's, used empty as an obvious flag for value being nan
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
        #if cnt == 15: ############### TEMP: For speeding up testing of updated code for main function delete once updates confirmed to work
        #    break ############### TEMP: For speeding up testing of updated code for main function delete once updates confirmed to work

    print("\t\t### ~ Started - Step 1 a): Agg Phase 1: Calculating Update Rates ~ ###") ############### TEMP: For Tracking test progress
    # Freeing up some memory
    temp_df = None
    weather_df = None
    # Calculating the update rate
    nc_data["update_rate"] = nc_data["count"] / cnt
    nc_data.drop("count", inplace=True, axis=1)

    #nc_data.to_csv('aggregated_no_units_data.csv') ############### TEMP: Write aggregation output to file to provide sample data for testing step 2, remove for final model

    # b) Encode and scale NC data
    # TODO: Look up the correct function name for fixing units of measurement (getting added to the query function, can remove/update to DONE once confirmed complete)
    # TODO: clean and correct units of measurement (getting added to the query function, can remove/update to DONE once confirmed complete)
    print("\t##### ~~~ Step 1 b): Started - Clustering Phase ~~~ #####") ############### TEMP: For Tracking test progress

    # Getting Indexes of the continuous columns
    cont_cols = [i for i in range(len(SENSOR_ID_TAGS),len(nc_data.columns))]

    # Scale Continuous
    scaled_data = data_preparation.scale_continuous(nc_data, cont_cols)
    nc_data = pd.concat([nc_data.iloc[:, 0:len(SENSOR_ID_TAGS)], pd.DataFrame(scaled_data, index=nc_data.index, columns=nc_data.columns[cont_cols].tolist())], axis=1)
    scaled_data = None
    
    # Encoding units
    nc_data = data_preparation.encode_units(nc_data)

    # c) cluster NC data to get df of sensor id fields + cluster group number
    # Calculating Gower's Distance, MDS, and clustering
    print("\t\t### ~ Started - Step 1 c): Clust Phase 1: Calculating Gower's Distance ~ ###") ############### TEMP: For Tracking test progress
    gow_dist = clustering.calc_gowers(nc_data, cont_cols)
    
    ###################################################################
    ############### NOTE: Doesn't look like we need MDS ###############
    ##########    keeping it here for now just incase the scaled up model
    ##########    performs worse without, but I don't see why it should
    #print("\t\t### ~ Started - Step 1 c): Clust Phase 2: Calculating MDS ~ ###") ############### TEMP: For Tracking test progress
    #mds_data = clustering.multidim_scale(gow_dist, num_dim=2)
    #clusters = AgglomerativeClustering(linkage = 'single', n_clusters=20).fit_predict(mds_data)
    ############### NOTE: Doesn't look like we need MDS ###############
    ###################################################################
    
    print("\t\t### ~ Started - Step 1 c): Clust Phase 3: Calculating Clusters ~ ###") ############### TEMP: For Tracking test progress
    clusters = AgglomerativeClustering(linkage = 'single', affinity='precomputed', n_clusters=20).fit_predict(gow_dist)

    # Generating a list of the columns to keep when making the dataframe relating sensors to clusters(the unique identifiers for an NC sensor and cluster)
    unique_cols_idx = [i for i in range(len(SENSOR_ID_TAGS))]
    unique_cols = nc_data.columns[unique_cols_idx].values.tolist()
    unique_cols.append("cluster")
    # Creating dataframe that identifies which unique sensors belong to which cluster
    drop_cols = list(set(nc_data.columns.tolist())-set(unique_cols))
    cluster_groups = pd.concat([nc_data, pd.DataFrame(clusters, columns=["cluster"])], axis=1)
    cluster_groups = cluster_groups.drop(drop_cols, axis=1)

    # d) Reload NC data + join cluster group num + aggregate, this time grouping by date, time, and clust_group_num
    print("\t##### ~~~ Started - Step 1 d): Aggregation Phase 2 ~~~ #####") ############### TEMP: For Tracking test progress
    last_idx_as_cols = True
    is_first_iter = True
    cnt=1
    for day in DATELIST:
        print("\t\t"+str(cnt)+": "+str(day)) ############### TEMP: For Tracking test progress
        # Querying and preping data for aggregations
        temp_df = data_preparation.query_csv(client=None, date=day, site=None)
        weather_df = data_preparation.query_weather_csv(client=None, date=day, site=None) ############### TEMP: To be replaced by actual query functions in final product
        if weather_df is None: ############### TEMP: To be replaced by actual query functions in final product
            pass
        else:
            temp_df = pd.concat([temp_df, weather_df]) ############### TEMP: To be replaced by actual query functions in final product
            temp_df = temp_df.fillna('empty') ############### TEMP: To be replaced by actual query functions in final product # Aggregation doesn't work with nan's, used empty as an obvious flag for value being nan
        
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
        #if cnt == 15: ############### TEMP: For speeding up testing of updated code for main function delete once updates confirmed to work
        #    break ############### TEMP: For speeding up testing of updated code for main function delete once updates confirmed to work

    print("\t\t### ~ Started - Step 1 d): Agg Phase 2: Calculating Update Rates ~ ###") ############### TEMP: For Tracking test progress
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
    print("####### ~~~~~ Starting - Step 2: Model EC/NC Relationship ~~~~~ #######") ############### TEMP: For Tracking test progress
    last_idx_as_cols = False
    is_first_iter = True
    cnt=1
    for day in DATELIST:
        # Querying and preping data for aggregations
        temp_df2 = data_preparation.query_csv(client=None, date=day, site=None)
        if temp_df2 is None:
            continue
        col_names = ['datetime']
        col_names.extend(temp_df2.columns[1:])
        temp_df2.columns = col_names
        temp_df2 = aggregation.split_datetime(temp_df2)
        # Filter for EC data, this step will be done in the query
        temp_df2=temp_df2[(temp_df2['unit']=='kWh') | (temp_df2['unit']=='m³')]
        # Creating uniqueId
        temp_df2=data_preparation.create_unique_id(temp_df2)
        # Filtering dataframe for only relevant fields
        temp_df2=temp_df2[['uniqueId', 'date', 'hour', 'unit', 'value']]
        if is_first_iter:
            # Creating a low memory dataframe for the append_agg function before the structure is changed by agg_all
            struct_df2 = temp_df2.head(1)
            # Aggregating the first date's data
            ec_data1=aggregation.agg_numeric_by_col(temp_df2, col_idx=[0,1,2,3], how='mean')
     ##### b) Also create second DF by aggregating further just using sensor ID fields (end result=1row per sensor)
            ec_data2=aggregation.agg_numeric_by_col(temp_df2, col_idx=[0,3], how='all')
            is_first_iter = False
        else:
            # Aggregating the current date's data and aggregate it with the current running total
            temp_df2a=aggregation.agg_numeric_by_col(temp_df2, col_idx=[0,1,2,3], how='mean')
            temp_df2b=aggregation.agg_numeric_by_col(temp_df2, col_idx=[0,3], how='all')
            ec_data1=aggregation.append_agg(df1=temp_df2a, df2=ec_data1, struct_df=struct_df2, col_idx=[0,1,2,3])
            ec_data2=aggregation.append_agg(df1=temp_df2b, df2=ec_data2, struct_df=struct_df2, col_idx=[0,3])
        cnt += 1
    # Freeing up some memory
    temp_df2 = None
    temp_df2a = None
    temp_df2b = None
    # Calculating the update rate
    ec_data2["update_rate"] = ec_data2["count"] / (cnt*24)
    ec_data2.drop("count", inplace=True, axis=1)

    # Resetting index columns
    ec_data1=ec_data1.reset_index()
    ec_data2=ec_data2.reset_index()

    # Renaming column
    ec_data1=ec_data1.rename(columns={"mean":"EC_mean_value"})

    # Dataframe with unique sensor ids
    uniqueSensors=ec_data2['uniqueId'].unique()

    ### Scaling EC data
    ec_data1['EC_mean_value']=data_preparation.scale_continuous(ec_data1, indexes=[4])


    ### Scaling Cluster data
    for i in range(6,len(nc_data.columns)):
        nc_data.iloc[:,i]=data_preparation.scale_continuous(nc_data, indexes=[i])

    #    c) For each unique EC sensorID (i.e. row in 2b_EC_data_df), create Ridge Regression model using 2a_EC_data_df and
    #       step1_output_NC_data_df. Model is basically: Y=EC response and Xn=NC data

    ### Will store each ridge output into a list and append all the dataframes
    coefficients_list=[]

    ### total sum of mse from each ridge regression model (accumulative)
    score=0

    ### Creating individual data frames for each sensor and implementing lasso
    for sensor in uniqueSensors:

        ## Create data frame for only that relevant sensor
        new_df=ec_data1[ec_data1['uniqueId']==sensor]
        ######## Changing EC data types for merging later. Might not need depending on step 1 output types
        nc_data = nc_data.astype({"date": str})
        new_df = new_df.astype({"date": object, "hour": object})
        new_df.loc[:,'date']=new_df['date'].apply(lambda x: str(x)[0:10])

        ## Merge specific sensor to cluster data
        new_merged=pd.merge(nc_data, new_df, how='inner', left_on=['date','hour'], right_on=['date','hour'])
        ## Ridge does not allow NANs, seems like some sensors are not 'on' during specific hours
        new_merged=new_merged.fillna(0)

        ## All NC predictor variables
        X=new_merged.iloc[:,2:(len(new_merged.columns)-3)]

        ## Mean value of EC data
        Y=new_merged['EC_mean_value']
        Y=Y.to_numpy().reshape(len(Y),1)

        #Ridge CV to find optimal alpha value
        alphas=[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
        reg=RidgeCV(alphas=alphas, store_cv_values=True)
        reg.fit(X, Y)
        alpha_best=reg.alpha_

      ### Ridge model using optimal alpha value found in step above
        ridge_test=Ridge(alpha=alpha_best, tol=.01, max_iter=10e7,normalize=True)
        ridge_test.fit(X,Y)
        coef=ridge_test.coef_
        mse=mean_squared_error(y_true=Y, y_pred=ridge_test.predict(X))
        score=score+mse

        ## Store coefficients into a dataframe
        new=pd.DataFrame(data=coef.reshape(1,((len(new_merged.columns)-3)-2)))

        ## Add uniqueId to the dataframe
        new['uniqueId']=sensor

        ## Store each sensorID's ridge coefficients into a list
        coefficients_list.append(new)

    ### Append all ridge coefficients for all sensors into a single dataframe
    for df in uniqueSensors:
        final_df = pd.concat(coefficients_list)

    ### calculate the avarge mse across all ridge regression models
    avg_mse=score/len(uniqueSensors)
    print("avg_mse:",avg_mse)

    #    OUTPUT OF STEP2 = dataframe with EC sensor ID fields, mean response, and all n coeffecients from
    #        that unique EC sensor's Ridge model
    print("####### ~~~~~ Complete - Step 2: Model EC/NC Relationship ~~~~~ #######") ############### TEMP: For Tracking test progress

    #### 3) Prep EC data for classification model
    ################################################
    print("####### ~~~~~ Starting - Step 3: Prep EC Data for Classification Model ~~~~~ #######") ############### TEMP: For Tracking test progress
    #### a) Load metadata and join with 2b_EC_data_df
    metadata=pd.read_csv('../data/PharmacyQuery.csv')
    # Make uniqueIDs
    metadata=data_preparation.create_unique_id(metadata, metadata=True)
    # Drop duplicates
    metadata=metadata.sort_values('lastSynced').drop_duplicates('uniqueId',keep='last')
    # Choose relevant fields
    metadata=metadata[['uniqueId','kind', 'energy','power', 'sensor', 'unit', 'water']]
    ### Changing boolean to easily identify during encoding process
    metadata['energy']=metadata['energy'].apply(lambda x: 'yes_energy' if x=='✓' else 'no_energy')
    metadata['power']=metadata['power'].apply(lambda x: 'yes_power' if x=='✓' else 'no_power')
    metadata['sensor']=metadata['sensor'].apply(lambda x: 'yes_sensor' if x=='✓' else 'no_sensor')
    metadata['water']=metadata['water'].apply(lambda x: 'yes_water' if x=='✓' else 'no_water')
    metadata['unit']=metadata['unit'].apply(lambda x: 'omit' if x=='_' else x)
    # left join metadata and 2b_EC_data_df
    merged_left=pd.merge(ec_data2, metadata, how='left', left_on='uniqueId', right_on='uniqueId')

    #### b) Apply feature selection function(s) to the joined EC+metadata
    # load NRCan classifications training data
    nrcan_labels=pd.read_csv('../data/FinalPharmacyECSensorList-WithLabels - PharmacyECSensorsWithLabels.csv')
    # make uniqueId
    nrcan_labels['siteRef']='Pharmacy'
    nrcan_labels=data_preparation.create_unique_id(nrcan_labels)

    # rename columns to fix unit of measurements
    nrcan_labels.rename(columns={'UBC_EWS.firstValue':'value'}, inplace=True)
    # run correct_df_units function
    nrcan_labels=data_preparation.correct_df_units(nrcan_labels)

    # TRAINING DATA CLEANING (maybe its own module with metadata?)
    # can change ? to 0 since uom fixed
    nrcan_labels=nrcan_labels.assign(isGas=nrcan_labels.isGas.apply(lambda x: '0' if x=='?' else x))
    # changing boolean for more descriptive encoding
    nrcan_labels=nrcan_labels.assign(isGas=nrcan_labels.isGas.apply(lambda x: 'no_gas' if x=='0' else 'yes_gas'))

    # selecting relevant training data fields
    nrcan_labels=nrcan_labels[['uniqueId', 'isGas', 'equipRef', 'groupRef', 'navName', 'endUseLabel']]
    nrcan_labels=nrcan_labels.drop_duplicates()
    merged_outer=pd.merge(left=merged_left, right=nrcan_labels, how='outer', left_on='uniqueId', right_on='uniqueId')
    # make equipRef and navName into smaller categories for feature engineering
    merged_outer=merged_outer.assign(equipRef=merged_outer.equipRef.apply(lambda x: data_preparation.equip_label(str(x))))
    merged_outer=merged_outer.assign(navName=merged_outer.navName.apply(lambda x: data_preparation.nav_label(str(x))))
    
    ##########################################################################
    ####################### CATEGORICAL FEATURE SELECTION ####################
    ##########################################################################
    categorical=merged_outer[['energy', 'power', 'sensor', 'unit', 'water', 'isGas', 'equipRef', 'navName', 'endUseLabel']].copy()
    categorical=categorical[(categorical['unit']=='kWh') | (categorical['unit']=='m³')]
    categorical=categorical.drop(['unit'], axis=1)

    ##########################################################################
    ###################### Preprocessing of Data #############################
    ##########################################################################
    #### Split data into training and test data
    dataset = categorical.values
    X = dataset[:, :-1]
    y = dataset[:,-1]
    X=X.astype(str)
    y=y.astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.42, random_state=1)

    #### Encode categorical data
    oe = OneHotEncoder(handle_unknown='ignore')
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)

    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    ##########################################################################
    ################### Done with Preprocessing of Data ######################
    ##########################################################################



    ##########################################################################
    ############# Choosing K=maxFeature through Cross-Validation #############
    ##########################################################################

    ### Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    ### The "accuracy" scoring is proportional to the number of correct classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
                  scoring='accuracy')
    rfecv.fit(X_train_enc, y_train_enc)

    ### Create dataframe with scroes and number of features
    Scores=pd.DataFrame({'scores':rfecv.grid_scores_,'num_features':range(1, len(rfecv.grid_scores_) + 1)})

    ### Choosing max number of features depending on how many features are recommended
    maxFeature=rfecv.n_features_

    print("Optimal number of features : %d" % rfecv.n_features_)

    ##### Plot Cross-Validation Scores and Number of Features
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.title("Cross-Validation Plot to Decide K (Categorical Data)")
    plt.show()    
    ##########################################################################
    ###################### Done with Cross-Validation ########################
    ##########################################################################



    ##########################################################################
    ###########  Mutual Information Technique using K=maxFeature #############
    ##########################################################################
    fs = SelectKBest(score_func=mutual_info_classif, k=int(maxFeature))
    fs.fit(X_train_enc, y_train_enc)
    X_train_fs = fs.transform(X_train_enc)
    X_test_fs = fs.transform(X_test_enc)

    ### Storing all feature labels
    all_feature_names=oe.get_feature_names()
    all_features_count=len(all_feature_names)

    ### Storing recommended feature labels
    fs.get_support(indices=True)
    feature_names = [all_feature_names[i] for i in fs.get_support(indices=True)]
    if feature_names:
        feature_names = np.asarray(feature_names)

    ### Modifying feature labels to include original field names
    new_feature_names=[]
    for i in range(len(categorical.columns)+1):
        if feature_names[i][1:2]==str(0):
            new_feature_names.append(feature_names[i].replace(feature_names[i][:2], categorical.columns[0]))
        elif feature_names[i][1:2]==str(1):
            new_feature_names.append(feature_names[i].replace(feature_names[i][:2], categorical.columns[1]))
        elif feature_names[i][1:2]==str(2):
            new_feature_names.append(feature_names[i].replace(feature_names[i][:2], categorical.columns[2]))
        elif feature_names[i][1:2]==str(3):
            new_feature_names.append(feature_names[i].replace(feature_names[i][:2], categorical.columns[3]))
        elif feature_names[i][1:2]==str(4):
            new_feature_names.append(feature_names[i].replace(feature_names[i][:2], categorical.columns[4]))
        elif feature_names[i][1:2]==str(5):
            new_feature_names.append(feature_names[i].replace(feature_names[i][:2], categorical.columns[5]))
        elif feature_names[i][1:2]==str(6):
            new_feature_names.append(feature_names[i].replace(feature_names[i][:2], categorical.columns[6]))
        elif feature_names[i][1:2]==str(7):
            new_feature_names.append(feature_names[i].replace(feature_names[i][:2], categorical.columns[7]))

    print('Original number of features: {:.0f}'.format(all_features_count))
    print('')
    print('These are the {:.0f} recommended categorical features:'.format(maxFeature))
    print('')
    print(new_feature_names)
    print('')

    ##########################################################################
    ##################### Done with Mutual Information #######################
    ##########################################################################
    
     ##########################################################################
    ######################## NUMERICAL FEATURE SELECTION #####################
    ##########################################################################

    ##########################################################################
    ###################### Preprocessing of Data #############################
    ##########################################################################
    numerical=merged_outer[['mean','std','max','min','update_rate','endUseLabel']]
    numerical=numerical.dropna(axis=0) # drop rows if they contain nan for enduselabel, because not part of training data
    dataset = numerical.values
    X = pd.DataFrame(dataset[:, :-1], columns=numerical.columns[0:5])
    y = dataset[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    ##########################################################################
    ################### Done with Preprocessing of Data ######################
    ##########################################################################


    ##########################################################################
    ################# Choosing K through Cross-Validation ####################
    ##########################################################################
    rfc = RandomForestClassifier(random_state=101)
    rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(2), scoring='accuracy')
    rfecv.fit(X_train, y_train)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.title("Cross-Validation Plot to Decide K (Numerical Data)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    ##########################################################################
    ###################### Done with Cross-Validation ########################
    ##########################################################################



    ##########################################################################
    #################  ANOVA F-Score Technique using K #######################
    ##########################################################################
    fvalue_selector = SelectKBest(f_classif, k=rfecv.n_features_)
    # Apply the SelectKBest object to the features and target
    X_kbest = fvalue_selector.fit_transform(X, y)
    # Create and fit selector
    cols = fvalue_selector.get_support(indices=True)
    numerical_features=X.iloc[:,cols].columns.tolist()
    # Show results
    print('Original number of features:', X.shape[1])
    print('')
    print('These are the {:.0f} recommended numerical features:'.format(X.shape[1]))
    print('')
    print(numerical_features)
    ##########################################################################
    ############################ Done with ANOVA #############################
    ##########################################################################

feature_selection()


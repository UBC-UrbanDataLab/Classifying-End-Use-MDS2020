#!/usr/bin/env python
# coding: utf-8

# In[73]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### ~ Library Imports ~ ###
# General Imports
from os import listdir
from os.path import isfile, join
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
# Modules Developed for this Project Imports
import data_preparation
import aggregation
import clustering
 
def feature_selection():
    #0) Set Constants
    ### ~ Required for base functionality ~ ###
    # String defining which site to run the model for
    QUERY_SITE = 'Pharmacy'
    # String defining the path to the metadata csv for the given building
    METADATA_CSV_PATH = '../data/PharmacyQuery.csv'
    # String defining the path to the training dataset
    TRAINING_SET_PATH = '../data/FinalPharmacyECSensorList-WithLabels - PharmacyECSensorsWithLabels.csv'
    # List of indices that can be combined to uniquely identify a sensor (used to group on each sensors)
    SENSOR_ID_TAGS = [1,2,3,4,5,6] # order is ["groupRef","equipRef","navName","siteRef","typeRef","unit"]
                                   # The planned update to the InfluxDB may change SENSOR_ID_TAGS to only [1] as in ["uniqueID"]
    ### ~ Alows customization of outputs ~ ###
    # Boolean defining if the output dataframes from each step should be saved (save if True, else False)
    SAVE_STEP_OUTPUTS = True
    # Boolean defining if the model should query from the database or pull from csv's (from database if True, else False)
    QUERY_FROM_DB = False
    # Strings containing the paths to the folders that contains the csv's to pull data from if QUERY_FROM_DB==False
    # All file names within the folders must be formatted as "YYYY-MM-DD.csv"
    QUERY_CSV_PATH = '../data/sensor_data/'
    QUERY_WEATHER_CSV_PATH = '../data/weather_data/'
        
    if QUERY_FROM_DB:
        # Getting a list of the last 90 dates or the list of date files to query from if QUERY_FROM_DB==False
        DATELIST =  [(date.today() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(1,91)]
        DATELIST.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d"))
    else:
        # Getting the list of files stored in the path provided in QUERY_CSV_PATH
        # All files names must be formatted as "YYYY-MM-DD.csv"
        DATELIST = [f.split(".")[0] for f in listdir(QUERY_CSV_PATH) if isfile(join(QUERY_CSV_PATH, f))]
    
    # Connecting to influxDB
    if QUERY_FROM_DB:
        client = data_preparation.connect_to_db()

    ##### 1) Cluster NC data
    ###########################################################################
    print("####### ~~~~~ Started - Step 1: Clustering Phase ~~~~~ #######") # For tracking program progress
    ###   a) load+aggregate NC data including weather, grouping by sensor ID fields
    print("\t##### ~~~ Started - Step 1 a): Aggregation Phase 1 ~~~ #####") # For tracking program progress
    last_idx_as_cols = False
    is_first_iter = True
    cnt=1
    for day in DATELIST:
        print("\t\t"+str(cnt)+": "+str(day)) # For tracking program progress
        # Querying and preping data for aggregations
        if QUERY_FROM_DB:
            temp_df = data_preparation.query_db_nc(client, day, num_days=1, site=QUERY_SITE)
            if temp_df is not None:
                # Making the datetime index into a column so that date and hour can be extracted later
                temp_df.reset_index(level=0, inplace=True)
        else:
            temp_df = data_preparation.query_csv(client=QUERY_CSV_PATH, date=day, site=None)
            weather_df = data_preparation.query_weather_csv(client=QUERY_WEATHER_CSV_PATH, date=day, site=None)
            if weather_df is None:
                pass
            else:
                temp_df = pd.concat([temp_df, weather_df])
                temp_df = temp_df.fillna('empty') # Aggregation doesn't work with nan's, using empty as an obvious flag for value being nan
        if temp_df is None:
            continue
        # Formatting the dataframe columns for date, month, and hour extraction
        col_names = ['datetime']
        col_names.extend(temp_df.columns[1:])
        temp_df.columns = col_names
        # Getting date, month, and hour
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
       
    print("\t\t### ~ Started - Step 1 a): Agg Phase 1: Calculating Update Rates ~ ###") # For tracking program progress
    # Freeing up some memory
    temp_df = None
    weather_df = None
    struct_df = None
    # Calculating the update rate
    nc_data["update_rate"] = nc_data["count"] / cnt
    nc_data.drop("count", inplace=True, axis=1)

    ###   b) Encode and scale NC data
    print("\t##### ~~~ Step 1 b): Started - Clustering Phase ~~~ #####") # For tracking program progress

    # Getting Indexes of the continuous columns
    cont_cols = [i for i in range(len(SENSOR_ID_TAGS),len(nc_data.columns))]

    # Scale Continuous
    scaled_data = data_preparation.scale_continuous(nc_data, cont_cols)
    nc_data = pd.concat([nc_data.iloc[:, 0:len(SENSOR_ID_TAGS)], pd.DataFrame(scaled_data, index=nc_data.index, columns=nc_data.columns[cont_cols].tolist())], axis=1)
    scaled_data = None
    
    # Encoding units
    nc_data = data_preparation.encode_units(nc_data)

    ###   c) cluster NC data to get df of sensor id fields + cluster group number
    # Calculating Gower's Distance, and clusters
    print("\t\t### ~ Started - Step 1 c): Clust Phase 1: Calculating Gower's Distance ~ ###") # For tracking program progress
    gow_dist = clustering.calc_gowers(nc_data, cont_cols)
    print("\t\t### ~ Started - Step 1 c): Clust Phase 2: Calculating Clusters ~ ###") # For tracking program progress
    clusters = AgglomerativeClustering(linkage = 'single', affinity='precomputed', n_clusters=20).fit_predict(gow_dist)
    # Freeing up some memory
    gow_dist = None
    
    # Generating a list of the columns to keep when making the dataframe relating sensors to clusters(the unique identifiers for an NC sensor and cluster)
    unique_cols_idx = [i for i in range(len(SENSOR_ID_TAGS))]
    unique_cols = nc_data.columns[unique_cols_idx].values.tolist()
    unique_cols.append("cluster")
    # Creating a dataframe that identifies which unique sensors belong to which cluster
    drop_cols = list(set(nc_data.columns.tolist())-set(unique_cols))
    cluster_groups = pd.concat([nc_data, pd.DataFrame(clusters, columns=["cluster"])], axis=1)
    cluster_groups = cluster_groups.drop(drop_cols, axis=1)

    ###   d) Reload NC data + join cluster group num + aggregate, this time grouping by date, time, and clust_group_num
    print("\t##### ~~~ Started - Step 1 d): Aggregation Phase 2 ~~~ #####") # For tracking program progress
    last_idx_as_cols = True
    is_first_iter = True
    cnt=1
    for day in DATELIST:
        print("\t\t"+str(cnt)+": "+str(day)) # For tracking program progress
        # Querying and preping data for aggregations
        if QUERY_FROM_DB:
            temp_df = data_preparation.query_db_nc(client, day, num_days=1, site=QUERY_SITE)
            if temp_df is not None:
                # Making the datetime index into a column so that date and hour can be extracted later
                temp_df.reset_index(level=0, inplace=True)
        else:
            temp_df = data_preparation.query_csv(client=QUERY_CSV_PATH, date=day, site=None)
            weather_df = data_preparation.query_weather_csv(client=QUERY_WEATHER_CSV_PATH, date=day, site=None)
            if weather_df is None:
                pass
            else:
                temp_df = pd.concat([temp_df, weather_df])
                temp_df = temp_df.fillna('empty') # Aggregation doesn't work with nan's, using empty as an obvious flag for value being nan
        if temp_df is None:
            continue
         # Formatting the dataframe columns for date, month, and hour extraction
        col_names = ['datetime']
        col_names.extend(temp_df.columns[1:])
        temp_df.columns = col_names
        # Getting date, month, and hour (must have at least hour for aggregations)
        temp_df = aggregation.split_datetime(temp_df)
        # Adding cluster groupings to the data for aggregation purposes
        temp_df = temp_df.merge(cluster_groups, how='left', on=cluster_groups.columns[:-1].tolist())
        if is_first_iter:
            # Calculating the count of sensor updates per hour per day per cluster for the first date
            update_rates = temp_df.groupby(['hour','date','cluster']).agg({'value':'count'},axis=1)
            # Creating a low memory dataframe for the append_agg function before the structure is changed by agg_all (different structure required than previous)
            struct_df = temp_df.head(1)
            # Identifying the indexes of the items being aggregated on (hour, date, and cluster)
            cluster_id_tags = [temp_df.columns.tolist().index("hour"), temp_df.columns.tolist().index("date"), temp_df.columns.tolist().index("cluster")]
            # Aggregating the first date's data (Aggregations must be seperate b/c data gets too big during calculations otherwise)
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
        
    print("\t\t### ~ Started - Step 1 d): Agg Phase 2: Calculating Update Rates ~ ###") # For tracking program progress
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
    if SAVE_STEP_OUTPUTS:
        nc_data.to_csv('step1_clustering_phase_output.csv')
    
    # Freeing up some space
    temp_df_aggs = None
    struct_df = None
    temp_df = None
    update_rates = None
    
    print("####### ~~~~~ Complete - Step 1: NC Aggregation and Clustering Phase ~~~~~ #######") # For tracking program progress

    ##### 2) Model EC/NC relationship
    ###########################################################################
    print("####### ~~~~~ Starting - Step 2: Model EC/NC Relationship ~~~~~ #######") # For tracking program progress
    ###   a) Aggregate EC sensors to be response variables for the regression model
    print("\t##### ~~~ Started - Step 2 a): Aggregation ~~~ #####") # For tracking program progress
    last_idx_as_cols = False
    is_first_iter = True
    cnt=1
    for day in DATELIST:
        print("\t\t"+str(cnt)+": "+str(day)) # For tracking program progress
        # Querying and preping data for aggregations
        if QUERY_FROM_DB:
            temp_df2 = data_preparation.query_db_ec(client, day, num_days=1, site=QUERY_SITE)
            if temp_df2 is not None:
                # Making the datetime index into a column so that date and hour can be extracted later
                temp_df2.reset_index(level=0, inplace=True)
        else:
            temp_df2 = data_preparation.query_csv(client=QUERY_CSV_PATH, date=day, site=None)
            # Filter for EC data, this step will be done in the query
            if temp_df2 is not None:
                temp_df2=temp_df2[(temp_df2['unit']=='kWh') | (temp_df2['unit']=='m³')]
        if temp_df2 is None:
            continue
        col_names = ['datetime']
        col_names.extend(temp_df2.columns[1:])
        temp_df2.columns = col_names
        temp_df2 = aggregation.split_datetime(temp_df2)
        # Creating uniqueId
        temp_df2=data_preparation.create_unique_id(temp_df2)
        # Filtering dataframe for only relevant fields
        temp_df2=temp_df2[['uniqueId', 'date', 'hour', 'unit', 'value']]
        if is_first_iter:
            # Creating a low memory dataframe for the append_agg function before the structure is changed by agg_all
            struct_df2 = temp_df2.head(1)
            # Aggregating the first date's data
            ec_data1=aggregation.agg_numeric_by_col(temp_df2, col_idx=[0,1,2,3], how='mean')
            # Also create second DF by aggregating further just using sensor ID fields (end result=1row per sensor)
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

    # Scaling EC data
    ec_data1['EC_mean_value']=data_preparation.scale_continuous(ec_data1, indexes=[4])


    # Scaling Cluster data
    for i in range(6,len(nc_data.columns)):
        nc_data.iloc[:,i]=data_preparation.scale_continuous(nc_data, indexes=[i])

    ###   b) For each unique EC sensorID (i.e. row in 2b_EC_data_df), create Ridge Regression model using
    #       2a_EC_data_df and step1_output_NC_data_df. Model is basically: Y=EC response and Xn=NC data
    print("\t##### ~~~ Started - Step 2 b): Regression ~~~ #####") # For tracking program progress

    # Will store each ridge output into a list and append all the dataframes
    coefficients_list=[]

    # total sum of mse from each ridge regression model (accumulative)
    score=0

    # Creating individual data frames for each sensor and implementing Ridge Regression
    for sensor in uniqueSensors:
        # Create data frame for only that relevant sensor
        new_df=ec_data1[ec_data1['uniqueId']==sensor]
        # Changing EC data types for merging later
        nc_data = nc_data.astype({"date": str})
        new_df = new_df.astype({"date": object, "hour": object})
        new_df.loc[:,'date']=new_df['date'].apply(lambda x: str(x)[0:10])

        # Merge specific sensor to cluster data
        new_merged=pd.merge(nc_data, new_df, how='inner', left_on=['date','hour'], right_on=['date','hour'])
        # Replacing NaN's with 0 (Ridge Regression doesn't allow NANs)
        new_merged=new_merged.fillna(0)

        # All NC predictor variables
        X=new_merged.iloc[:,2:(len(new_merged.columns)-3)]

        # Mean value of EC data
        Y=new_merged['EC_mean_value']
        Y=Y.to_numpy().reshape(len(Y),1)

        # Ridge CV to find optimal alpha value
        alphas=[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
        reg=RidgeCV(alphas=alphas, store_cv_values=True)
        reg.fit(X, Y)
        alpha_best=reg.alpha_

        # Ridge model using optimal alpha value found in step above
        ridge_test=Ridge(alpha=alpha_best, tol=.01, max_iter=10e7,normalize=True)
        ridge_test.fit(X,Y)
        coef=ridge_test.coef_
        mse=mean_squared_error(y_true=Y, y_pred=ridge_test.predict(X))
        score=score+mse

        # Store coefficients into a dataframe
        new=pd.DataFrame(data=coef.reshape(1,((len(new_merged.columns)-3)-2)))

        # Add uniqueId to the dataframe
        new['uniqueId']=sensor

        # Store each sensorID's ridge coefficients into a list
        coefficients_list.append(new)

    # Append all ridge coefficients for all sensors into a single dataframe
    for df in uniqueSensors:
        final_df = pd.concat(coefficients_list)

    # calculate the avarge mse across all ridge regression models
    avg_mse=score/len(uniqueSensors)
    print("avg_mse:",avg_mse)
    
    if SAVE_STEP_OUTPUTS:
        pass # TODO: I'm not sure what data should be getting save here
    print("####### ~~~~~ Complete - Step 2: Model EC/NC Relationship ~~~~~ #######") # For tracking program progress

    ##### 3) Prep EC data for classification model
    ###########################################################################
    print("####### ~~~~~ Starting - Step 3: Prep EC Data for Classification Model ~~~~~ #######") # For tracking program progress
    ###   a) Load metadata and join with 2b_EC_data_df
    metadata=pd.read_csv(METADATA_CSV_PATH, dtype=object)
    # Make uniqueIDs
    metadata=data_preparation.create_unique_id(metadata, metadata=True)
    # Drop duplicates
    metadata=metadata.sort_values('lastSynced').drop_duplicates('uniqueId',keep='last')
    # Choose relevant fields
    metadata=metadata[['uniqueId','kind', 'energy','power', 'sensor', 'unit', 'water']]
    # Changing boolean to easily identify during encoding process
    metadata['energy']=metadata['energy'].apply(lambda x: 'yes_energy' if x=='✓' else 'no_energy')
    metadata['power']=metadata['power'].apply(lambda x: 'yes_power' if x=='✓' else 'no_power')
    metadata['sensor']=metadata['sensor'].apply(lambda x: 'yes_sensor' if x=='✓' else 'no_sensor')
    metadata['water']=metadata['water'].apply(lambda x: 'yes_water' if x=='✓' else 'no_water')
    metadata['unit']=metadata['unit'].apply(lambda x: 'omit' if x=='_' else x)
    # left join metadata and 2b_EC_data_df
    merged_left=pd.merge(ec_data2, metadata, how='left', left_on='uniqueId', right_on='uniqueId')

    ###   b) Apply feature selection function(s) to the joined EC+metadata
    # load NRCan classifications training data
    nrcan_labels=pd.read_csv(TRAINING_SET_PATH)
    # make uniqueId
    nrcan_labels['siteRef']=QUERY_SITE
    nrcan_labels=data_preparation.create_unique_id(nrcan_labels)

    # rename columns to fix unit of measurements
    nrcan_labels.rename(columns={'UBC_EWS.firstValue':'value'}, inplace=True)
    # run correct_df_units function
    nrcan_labels=data_preparation.correct_df_units(nrcan_labels)

    # TRAINING DATA CLEANING
    # Change ? to 0 since uom fixed
    nrcan_labels=nrcan_labels.assign(isGas=nrcan_labels.isGas.apply(lambda x: '0' if x=='?' else x))
    # changing boolean for more descriptive encoding
    nrcan_labels=nrcan_labels.assign(isGas=nrcan_labels.isGas.apply(lambda x: 'no_gas' if x=='0' else 'yes_gas'))

    # selecting relevant training data fields
    nrcan_labels=nrcan_labels[['uniqueId', 'isGas', 'equipRef', 'groupRef', 'navName', 'endUseLabel']]
    nrcan_labels=nrcan_labels.drop_duplicates()
    merged_outer=pd.merge(left=merged_left, right=nrcan_labels, how='left', left_on='uniqueId', right_on='uniqueId')
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
    ############# Choosing K=max_feature through Cross-Validation #############
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
    max_feature=rfecv.n_features_

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
    ###########  Mutual Information Technique using K=max_feature #############
    ##########################################################################
    fs = SelectKBest(score_func=mutual_info_classif, k=int(max_feature))
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
    print('These are the {:.0f} recommended categorical features:'.format(max_feature))
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


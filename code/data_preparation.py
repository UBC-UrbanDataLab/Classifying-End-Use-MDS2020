#!/usr/bin/env python
# coding: utf-8

# # File for Testing the Functions and Showing how to call them

# ***
## Data Preparation

# Library Imports
# Data storing Imports
import numpy as np
import pandas as pd

# Feature Engieering Imports
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Function to get Datatype of a value
def get_data_type(x):
    try:
        if x=='True' or x=='False':
            return 'bool'
        else:
            float(x)
            return 'num'
    except:
        return 'str'

# Creating seperate dataframes for categorical values and continuous values
def seperate_cat_and_cont(df, idx=0):
    df = df.copy()
    df['dtype'] = df.iloc[:,idx].apply(lambda x: get_data_type(x))
    cat_df = df[df['dtype']!='num']
    cont_df = df[df['dtype']=='num']
    return cat_df, cont_df

# Encoding Categorical Data
def encode_categorical(df, indexes = [0]):
    df = df.copy()
    isFirst = True
    for idx in indexes:
        unit2idx = dict(map(reversed,pd.DataFrame(df.iloc[:,idx].unique()).to_dict()[0].items()))
        df.iloc[:,idx] = df.iloc[:,idx].apply(lambda x: unit2idx[x])
        encoder = OneHotEncoder(handle_unknown='ignore')
        encodedUnits = encoder.fit_transform(np.reshape(df.iloc[:,idx].to_numpy(),(-1,1))).toarray()
        if isFirst:
            np_arr = encodedUnits
            isFirst = False
        else:
            np_arr = np.append(np_arr, encodedUnits,axis=1)
    return np_arr

# Scaling Continuous Data
def scale_continuous(df, indexes=[0]):
    isFirst = True
    for idx in indexes:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(np.reshape(df.iloc[:,idx].to_numpy(),(-1,1)))
        if isFirst:
            np_arr = scaled_data
            isFirst = False
        else:
            np_arr = np.append(np_arr, scaled_data ,axis=1)
    return np_arr

# Function to Encode and Scale values, outputs a dataframe with a scaled values column, and seperate dummy variable columns for each category option
def encode_and_scale_values(df):
    df = df.copy()
    # Generate seperate dataframes for categorical and continous data
    cat_data, cont_data = seperate_cat_and_cont(df,7)
    
    # Encode Data
    encode_catVals = encode_categorical(cat_data,[7])
    # Creating dataframe for storing encoded values
    encoded_df = pd.concat([cat_data, pd.DataFrame(encode_catVals, index=cat_data.index, columns=[str(i) for i in range(len(encode_catVals[0]))])], axis=1)
    encoded_df = encoded_df.add_prefix('cv_')
    # Drop Duplicated Columns
    drop_cols = encoded_df.columns
    drop_cols = drop_cols[0:len(drop_cols)-len(encode_catVals[0])]
    # Add encoded data columns to the original dataframe
    encoded_df = pd.concat([df, encoded_df], axis=1)
    encoded_df = encoded_df.drop(columns=drop_cols)
    
    # Scale Data
    scaling_contVals = scale_continuous(cont_data, [7])
    # Creating dataframe for storing scaled values
    scaled_df = pd.concat([cont_data, pd.DataFrame(scaling_contVals, index=cont_data.index, columns=[str(i) for i in range(len(scaling_contVals[0]))])], axis=1)
    scaled_df = scaled_df.add_prefix('sc_')
    # Drop Duplicated Columns
    drop_cols = scaled_df.columns
    drop_cols = drop_cols[0:len(drop_cols)-len(scaling_contVals[0])]
    # Add scaled data columns to the combined encoded data and original data dataframe
    encoded_and_scaled_df = pd.concat([encoded_df, scaled_df], axis=1)
    encoded_and_scaled_df = encoded_and_scaled_df.drop(columns=drop_cols)
    encoded_and_scaled_df = encoded_and_scaled_df.fillna(-1)
    return encoded_and_scaled_df

# Function to encode units
def encode_units(df):
    df = df.copy()
    encoded_units = encode_categorical(df,[6])
    # Creating dataframe for storing encoded units
    encoded_units_df = pd.concat([df, pd.DataFrame(encoded_units, index=df.index, columns=[str(i) for i in range(len(encoded_units[0]))])], axis=1)
    encoded_units_df = encoded_units_df.add_prefix('unit_')
    return encoded_units_df

# ***
# # Database Connection and Querying
# Library Imports for Influx Queries
import influxdb
from datetime import timezone, datetime
import pytz
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import certifi
import time

# Function to connect to the database
def connect_to_db(database = 'SKYSPARK'):
    # Options for database are SKYSPARK and ION, default is SKYSPARK
    client = influxdb.DataFrameClient(host='206.12.92.81',port=8086, 
                                      username='public', password='public',database=database)
    try:
        client.ping()
        print("Successful Connection")
        return client
    except:
        print("Failure to Connect")

# Funciton to check connection to the database
def check_connection(client):
    try:
        client.ping()
        print("Connected")
        return True
    except:
        print("Disconnected")
        return False

# Function to query a date range fromt he database
def query_db(client, date, num_days=1, site='Pharmacy'):
    start_date = date
    for i in range(0,num_days):
        print(date)
        time1 = '00:00:00'
        time2 = '23:59:59'
        query = 'select * from UBC_EWS where siteRef=$siteRef and time > $time1 and time < $time2'
        where_params = {'siteRef': site, 'time1':date+' '+time1, 'time2':date+' '+time2}
        result = client.query(query = query, bind_params = where_params, chunked=True, chunk_size=10000)
        if i==0:
            df=result['UBC_EWS']
        else:
            df=pd.concat([df,result['UBC_EWS']],axis=0)
            time.sleep(5)
    try:
        print("Time zone in InfluxDB:",df.index.tz)
        my_timezone = pytz.timezone('Canada/Pacific')
        df.index=df.index.tz_convert(my_timezone)
        print("Converted to",my_timezone,"in dataframe")
        print("Dataframe memory usage in bytes:",f"{df.memory_usage().values.sum():,d}")
        return df
    except:
        print("No data found for specified query")
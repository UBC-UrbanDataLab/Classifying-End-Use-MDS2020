#!/usr/bin/env python
# coding: utf-8

# # File for Testing the Functions and Showing how to call them

# ***
## Data Preparation

# Library Imports

import re

# Data storing Imports
import numpy as np
import pandas as pd

# Influx Imports
import influxdb
import pytz
import time

# Feature Engieering Imports
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Data Preparation Functions
def get_data_type(x):
    """Function to identify the value type of the string passed in
    Args:
        x (str): string format of data that is either a boolean, numeric value, or string
        
    Returns:
        (str): a string representing the value type of the string passed into the function 'bool', 'num', or 'str'
    """
    try:
        if x=='True' or x=='False' or x==True or x==False:
            return 'bool'
        else:
            float(x)
            return 'num'
    except:
        return 'str'


def separate_cat_and_cont(df, idx=0):
    """Function to separate continuous and categorical data into separate dataframes
    Args:
        df (pandas.DataFrame): dataframe to containing a column of mixed categorical and continuous data
        idx (int): index of the column containing mixed categorical and continuous data
        
    Returns:
        cat_df (pandas.DataFrame): original dataframe filterd down to only include observations with categorical values
        cont_df (pandas.DataFrame): original dataframe filterd down to only include observations with continuous values
    """
    df = df.copy()
    df['dtype'] = df.iloc[:,idx].apply(lambda x: get_data_type(x))
    cat_df = df[df['dtype']!='num']
    cont_df = df[df['dtype']=='num']
    return cat_df, cont_df

def encode_categorical(df, indexes = [0]):
    """Function to encode categorical data, the user must define which columns have categorical data
    Args:
        df (pandas.DataFrame): dataframe to containing at least one column of categorical data
        indexes (list): list of indexes with categorical data to encode
        
    Returns:
        np_arr (numpy.array): numpy array of encode categorical values
    """
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

def scale_continuous(df, indexes=[0]):
    """Function to scale continuous data, the user must define which columns have continuous data
    Args:
        df (pandas.DataFrame): dataframe to containing at least one column of continuous data
        indexes (list): list of indexes with continuous data to scale
        
    Returns:
        np_arr (numpy.array): numpy array of scaled continuous values
    """
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


def encode_and_scale_values(df):
    """Function to encode and scale values, outputs a dataframe with a scaled values column, and separate dummy variable columns for each category option
    Args:
        df (pandas.DataFrame): dataframe to containing a "value" column
        
    Returns:
        encoded_units_df (pandas.DataFrame): dataframe containing scaled numeric values, and encoded categorical values 
                                             (with appropriate dummy variables)
    """
    df = df.copy()
    # Generate separate dataframes for categorical and continous data
    cat_data, cont_data = separate_cat_and_cont(df,7)
    
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
    encoded_and_scaled_df = encoded_and_scaled_df.fillna(0)
    return encoded_and_scaled_df

def encode_units(df):
    """Function to encode units
    Args:
        df (pandas.DataFrame): dataframe to containing a "unit" column
        
    Returns:
        encoded_units_df (pandas.DataFrame): dataframe containing encoded units
    """
    df = df.copy()
    encoded_units = encode_categorical(df, [df.columns.tolist().index('unit')])
    # Creating dataframe for storing encoded units
    encoded_units_df = pd.concat([df, pd.DataFrame(encoded_units, index=df.index, columns=[str(i) for i in range(len(encoded_units[0]))]).add_prefix('unit_')], axis=1)
    return encoded_units_df

# ***
# # Database Connection and Querying

def connect_to_db(database = 'SKYSPARK'):
    """Function to connect to the database
    Args:
        database (string): name of the database to connect to options are 'SKYSPARK' (default) and 'ION'
        
    Returns:
        client (influxdb-python client object): database connection object
        OR
        None: If the database connection failed
    """
    client = influxdb.DataFrameClient(host='206.12.92.81',port=8086, 
                                      username='public', password='public',database=database)
    try:
        client.ping()
        print("Successful Connection")
        return client
    except:
        print("Failure to Connect")
        return None

def check_connection(client):
    """Funciton to check connection to the database

    Args:
        client (influxdb-python client object): database connection object
        
    Returns:
        True: If the database connection is live
        OR
        False: If the database connection is not live
    """
    try:
        client.ping()
        print("Connected")
        return True
    except:
        print("Disconnected")
        return False

def query_db(client, date, num_days=1, site='Pharmacy'):
    """Function to query the UBC_EWS database for the user defined start date, number of days (default=1), and site (default=Pharmacy)

    Args:
        client (influxdb-python client object): database connection object
        date (string): date of interest in format 'YYYY-MM-DD' such as '2020-05-05'
        num_days (int): number of days to query data
        site (string): name of builing of interest
        
    Returns:
        pandas.DataFrame: data from the queried date(s)
        OR
        None: return value if the query found no data
    """
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
        return None

def query_csv(client, date, site):
    """Function to read the csv of saved data from the influxDB for the specified date. Requires csv files
    to already be saved in a test_data subfolder. This is a temporary function to make testing faster while
    developing code. It is meant to be replaced with query_db() so that the project will actually pull data
    directly from the database.

    Args:
        date (string): date of interest in format 'YYYY-MM-DD' such as '2020-05-05'
        
    Dummy Args:
        site (string): name of builing of interest. Not actually used but will make it easier to replace
                        this function with the proper query_db() function in main
        client (influxdb-python client object): database connection object. Not actually used, kept as a placeholder
                        to make it easier to replace query_csv with query_db() function when it comes time to do so
                        in the main() function.
    Returns:
        pandas.DataFrame: contents of the specific csv
        OR
        None: couldn't find/access the specified csv

    """
    regexp = re.compile(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}')
    if not regexp.search(date):
        raise ValueError("Date was not entered in usable format: YYYY-MM-DD")
    try:
        filename = date+".csv"
        temp_df = pd.read_csv("test_data/"+filename)
        return temp_df
    except ValueError as e:
        print("ERROR: ", e)
        return None
    except OSError as e:
        print("ERROR: Unable to find or access file:", e)
        return None
    except:
        return None

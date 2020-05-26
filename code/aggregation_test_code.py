#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:06:05 2020

@author: connor
"""
# Library Imports
import pandas as pd

# Function imports from other files
import data_preparation as dp

# Code to Aggregate numeric data
data = pd.read_csv('test_data/2020-05-01.csv')
data = data.append(pd.read_csv('test_data/2020-03-16.csv'))

# Function to create and populate columns for the date and the hour of the day of each observation
def get_date_and_hour(df):
    # Takes a raw dataframe (no pre-processing after querying data)
    df = df.copy()
    # Extracting the date and hour of the day from the timestamp column
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour.astype(str) # Want as str so that it doesn't get aggregated when not aggregating on it
    return df

# Testing get_date_and_hour() function
data = get_date_and_hour(data)

# A function to aggregate the numeric data by the specified columns in a user defined manner
def agg_numeric_by_col(df, col_idx, how='mean'):
    # Takes a dataframe to aggregate numeric data for and the columns to aggregate on
    df = df.copy()
    # Filtering down just to the numeric values
    df['dtype'] = df['value'].apply(dp.get_data_type)
    df = df.loc[df['dtype']=='num']
    # Converting value column to float
    df['value'] = df['value'].astype(float)
    # Get column names to aggregate by
    groupNames = df.columns[col_idx].values.tolist()
    if how=='mean':
        df_agg = df.groupby(groupNames).mean()
    elif how=='median':
        df_agg = df.groupby(groupNames).median()
    elif how=='std':
        df_agg = df.groupby(groupNames).std()
    elif how=='max':
        df_agg = df.groupby(groupNames).max()[['value']] # Just get from the value column
    elif how=='min':
        df_agg = df.groupby(groupNames).min()[['value']] # Just get from the value column
    else:
        print('Invalid how argument, only capable of mean, std, max, or min.')
        return None
    # Aggregate and return values
    return df_agg

# Testing aggregating on numeric function using unique ID columns and hour of the day, and how=mean
print(agg_numeric_by_col(data, [1,2,3,4,5,9], 'mean'))
# Testing aggregating on numeric function using unique ID columns and date, and how=median
print(agg_numeric_by_col(data, [1,2,3,4,5,8], 'median'))
# Testing aggregating on numeric function using unit column and hour of the day, and how=std
print(agg_numeric_by_col(data, [6,9], 'std'))
# Testing aggregating on numeric function using unit columns and date, and how=max
print(agg_numeric_by_col(data, [6,8], 'max'))
# Testing aggregating on numeric function using unit columns and date, and how=min
print(agg_numeric_by_col(data, [6,8], 'min'))

# For easily viewing the data (Delete when done)
test = data.sample(20)


# Function to provide a count of the number boolean value changes grouped by the specified columns
def agg_bool_by_col(df, col_idx):
    # Takes a dataframe to aggregate boolean data for and the columns to aggregate on
    df = data.copy()
    # Filtering down just to the boolean values
    df['dtype'] = df['value'].apply(dp.get_data_type)
    df = df.loc[df['dtype']=='bool']
    # Converting value column to 0 (False) and 1 (True)
    df['value'] = df['value'].apply(lambda x: 1 if x=='True' else 0)
    # Converting indexes to group by to the column names
    groupNames = df.columns[col_idx].values.tolist()
    # Sorting observations by their timestamp
    df = df.sort_values(by='datetime')
    # Creating column for easy aggregation
    df['groups'] = ''
    for idx in col_idx:
            df.loc[:,'groups'] += df.iloc[:,idx].astype(str)
    isFirst = True
    groupList = df.loc[:,'groups'].unique()
    # For loop to identify the number of changes for each group individually
    for group in groupList:
        temp_df = df[df.loc[:,'groups']==group]
        temp_df.loc[:,'isChanged'] = temp_df.loc[:,'value'].diff().abs() # If value changes then will be 1, else will be 0
        temp_df = temp_df.drop(labels='groups', axis=1)
        temp_df['isChanged'] = temp_df.loc[:,'isChanged'].fillna(0)
        temp_df = temp_df.groupby(groupNames).sum()['isChanged']
        if isFirst==True:
                return_df = temp_df
                isFirst = False
        else:
            return_df = return_df.append(temp_df)
    return return_df

# Testing agg_bool_by_col() function
test_bool_agg = agg_bool_by_col(data, [1,2,3,4,5,9])

# Function to provide a count of the number categorical value changes grouped by the specified columns
def agg_cat_by_col(df, col_idx):
    # Takes a dataframe to aggregate boolean data for and the columns to aggregate on
    df = data.copy()
    # Filtering down just to the categorical values
    df['dtype'] = df['value'].apply(dp.get_data_type)
    df = df.loc[df['dtype']=='str']
    # Converting value column numeric indexes related to unique states in the value column
    unit2idx = dict(map(reversed,pd.DataFrame(df['value'].unique()).to_dict()[0].items()))
    df['value'] = df['value'].apply(lambda x: unit2idx[x])
    # Converting indexes to group by to the column names
    groupNames = df.columns[col_idx].values.tolist()
    # Sorting observations by their timestamp
    df = df.sort_values(by='datetime')
    # Creating column for easy aggregation
    df['groups'] = ''
    for idx in col_idx:
            df.loc[:,'groups'] += df.iloc[:,idx].astype(str)
    isFirst = True
    groupList = df.loc[:,'groups'].unique()
    # For loop to identify the number of changes for each group individually
    for group in groupList:
        temp_df = df[df.loc[:,'groups']==group]
        temp_df.loc[:,'isChanged'] = temp_df.loc[:,'value'].diff().abs()
        temp_df = temp_df.drop(labels='groups', axis=1)
        temp_df['isChanged'] = temp_df.loc[:,'isChanged'].fillna(0)
        temp_df['isChanged'] = temp_df['isChanged'].apply(lambda x: 1 if x>0 else 0)
        temp_df = temp_df.groupby(groupNames).sum()['isChanged']
        if isFirst==True:
                return_df = temp_df
                isFirst = False
        else:
            return_df = return_df.append(temp_df)
    return return_df

# Testing agg_bool_by_col() function
test_cat_agg = agg_cat_by_col(data, [1,2,3,4,5,8])

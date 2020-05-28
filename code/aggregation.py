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

# Function to create and populate columns for the date, month, and the hour of the day of each observation
def split_datetime(df):
    # Takes a raw dataframe (no pre-processing after querying data)
    df = df.copy()
    # Extracting the date, month, and hour of the day from the timestamp column
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['month'] = df['datetime'].dt.month.astype(str)
    df['hour'] = df['datetime'].dt.hour.astype(str) # Want as str so that it doesn't get aggregated when not aggregating on it
    return df

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

# Function to provide a count of the number boolean value changes grouped by the specified columns
def agg_bool_by_col(df, col_idx):
    # Takes a dataframe to aggregate boolean data for and the columns to aggregate on
    df = df.copy()
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
        temp_df.loc[:,'value'] = temp_df.loc[:,'value'].diff().abs() # If value changes then will be 1, else will be 0
        temp_df = temp_df.drop(labels='groups', axis=1)
        temp_df['value'] = temp_df.loc[:,'value'].fillna(0)
        temp_df = temp_df.groupby(groupNames).sum()['value'].to_frame()
        if isFirst==True:
                return_df = temp_df
                isFirst = False
        else:
            return_df = return_df.append(temp_df)
    return return_df

# Function to provide a count of the number categorical value changes grouped by the specified columns
def agg_cat_by_col(df, col_idx):
    # Takes a dataframe to aggregate boolean data for and the columns to aggregate on
    df = df.copy()
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
        temp_df.loc[:,'value'] = temp_df.loc[:,'value'].diff().abs()
        temp_df = temp_df.drop(labels='groups', axis=1)
        temp_df['value'] = temp_df.loc[:,'value'].fillna(0)
        temp_df['value'] = temp_df['value'].apply(lambda x: 1 if x>0 else 0)
        temp_df = temp_df.groupby(groupNames).sum()['value'].to_frame()
        if isFirst==True:
                return_df = temp_df
                isFirst = False
        else:
            return_df = return_df.append(temp_df)
    return return_df

# Function to fun all three aggregation types covered by the above functions and output data columnwise for input into other functions (ex. feature selection, and scaling)
def agg_all(df, col_idx, num_how='mean', last_idx_to_col=True):
    # Takes a dataframe to aggregate, columns to aggregate on, and how to aggregate the numeric values (bool and cat only have one type of aggregation), and if the last index should become columns (default=True)
    # Aggregating each datatype
    num_agg = agg_numeric_by_col(df, col_idx, how=num_how)
    bool_agg = agg_bool_by_col(df, col_idx)
    cat_agg = agg_cat_by_col(df, col_idx)
    # Combining all aggregations into one dataframe
    all_agg = num_agg.append(bool_agg)
    all_agg = all_agg.append(cat_agg)
    if last_idx_to_col == True:
        groups = all_agg.reset_index().iloc[:,-2].unique()
        new_df = all_agg.reset_index().drop(all_agg.reset_index().columns[-2:].tolist(), axis=1).drop_duplicates()
        onList = new_df.columns.tolist()
        for group in groups:
            cur_df = all_agg.reset_index()[all_agg.reset_index().iloc[:,-2]==group].drop(all_agg.reset_index().columns[-2], axis=1)
            new_df = pd.merge(new_df, cur_df, on=onList, how='left')
            new_df = new_df.rename(columns={'value':group})
        all_agg = new_df.fillna(0)
    else:
        all_agg = all_agg.reset_index().fillna(0)
    return all_agg
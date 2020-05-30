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
def split_datetime(df):
    """Function to create and populate columns for the date and hour value from the timestamp of each observation

    Args:
        df (pandas.DataFrame): dataframe containing a 'datetime' column
        
    Returns:
        pandas.DataFrame: input dataframe with 'date', 'month', 'hour' columns appended

    """
    # Takes a raw dataframe (no pre-processing after querying data)
    df = df.copy()
    # Extracting the date and hour of the day from the timestamp column
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['month'] = df['datetime'].dt.month.astype(str)
    df['hour'] = df['datetime'].dt.hour.astype(str) # Want as str so that it doesn't get aggregated when not aggregating on it
    return df

# Testing split_datetime() function
data = split_datetime(data)

# A function to aggregate the numeric data by the specified columns in a user defined manner
def agg_numeric_by_col(df, col_idx, how='all'):
    """Function to aggregate numeric data in user specified column using specified aggregation function

    Args:
        df (pandas.DataFrame): dataframe containing at least a numeric 'value' column
        col_idx (list): one or more column indicies to group by when aggregating
        how (str): 'mean', 'median', 'std', 'max', 'min'
        
    Returns:
        pandas.DataFrame: Dataframe with col_idx columns as the indexes and the appropriate aggregation(s) as columns
    """
    # Takes a dataframe to aggregate numeric data for and the columns to aggregate on
    df = df.copy()
    # Filtering down just to the numeric values
    df['dtype'] = df['value'].apply(dp.get_data_type)
    df = df.loc[df['dtype']=='num']
    # Converting value column to float
    df['value'] = df['value'].astype(float)
    # Get column names to aggregate by
    group_names = df.columns[col_idx].values.tolist()
    if how=='all':
        df_agg = df.groupby(group_names).agg({'value':['mean','std','max','min','count']},axis=1)
    else:
        try:
            df_agg = df.groupby(group_names).agg({'value':[how,'count']},axis=1)
        except:
            print('Invalid how argument, only capable of mean, std, max, or min.')
            return None
    # Drop unwanted levels and return values
    df_agg.columns = df_agg.columns.droplevel()
    return df_agg.fillna(0)

# Testing aggregating on numeric function using unique ID columns and month, and how=mean
test_num1 = agg_numeric_by_col(data, [1,2,3,4,5,9], 'mean')
print(test_num1)
# Testing aggregating on numeric function using unique ID columns and date, and how=all
test_num2 = agg_numeric_by_col(data, [1,2,3,4,5,8], 'all')
print(test_num2)
# Testing aggregating on numeric function using unit column and hour of the day, and how=std
test_num3 = agg_numeric_by_col(data, [6,10], 'std')
print(test_num3)
# Testing aggregating on numeric function using unit columns and month, and how=max
test_num4 = agg_numeric_by_col(data, [6,9], 'max')
print(test_num4)
# Testing aggregating on numeric function using unit columns and date, and how=min
test_num5 = agg_numeric_by_col(data, [6,8], 'min')
print(test_num5)

# Function to provide a count of the number boolean value changes grouped by the specified columns
def agg_bool_by_col(df, col_idx, how='all'):
    # Takes a dataframe to aggregate boolean data for and the columns to aggregate on
    df = df.copy()
    # Filtering down just to the boolean values
    df['dtype'] = df['value'].apply(dp.get_data_type)
    df = df.loc[df['dtype']=='bool']
    # Converting value column to 0 (False) and 1 (True)
    df['value'] = df['value'].apply(lambda x: 1 if x=='True' else 0)
    # Converting indexes to group by to the column names
    group_names = df.columns[col_idx].values.tolist()
    group_names.extend(['hour']) # First aggregation is by hour
    # Sorting observations by their timestamp
    df = df.sort_values(by='datetime')
    # Creating column for easy aggregation (always aggregate by hour first so do hour first)
    df['groups'] = ''
    for idx in col_idx:
            df.loc[:,'groups'] += df.iloc[:,idx].astype(str)
    # Adding hr_idx to the end of groups for initial grouping on hours
    isFirst = True
    groupList = df.loc[:,'groups'].unique()
    # For loop to identify the number of changes for each group individually
    for group in groupList:
        temp_df = df[df.loc[:,'groups']==group]
        temp_df.loc[:,'value'] = temp_df.loc[:,'value'].diff().abs() # If value changes then will be 1, else will be 0
        temp_df = temp_df.drop(labels='groups', axis=1)
        temp_df.loc[:,'value'] = temp_df.loc[:,'value'].fillna(0)
        temp_df = temp_df.groupby(group_names).agg({'value':['sum','count']},axis=1)
        if isFirst==True:
                return_df = temp_df
                isFirst = False
        else:
            return_df = return_df.append(temp_df)
    group_names = group_names[:-1] # Drop the added hours value to 
    return_df.columns = return_df.columns.droplevel()
    if how=='all':
        return_df = return_df.groupby(group_names).agg({'sum':['mean','std','max','min'], 'count':'sum'},axis=1) # Sum b/c the hours aggregation did the counts
        return_df.columns = return_df.columns.droplevel()
        return_df= return_df.rename(columns={'sum':'count'})
    else:
        return_df = return_df.groupby(group_names).agg({'sum':how,'count':'sum'},axis=1) # Sum b/c the hours aggregation did the counts
        if type(how)==list:
            return_df.columns = return_df.columns.droplevel()
            return_df= return_df.rename(columns={'sum':'count'})
        else:
            return_df= return_df.rename(columns={'sum':how})
    return return_df.fillna(0)

# Testing agg_bool_by_col() function
test_bool_agg = agg_bool_by_col(data, [1,2,3,4,5,8], ['mean','std'])

# Function to provide a count of the number categorical value changes grouped by the specified columns
def agg_cat_by_col(df, col_idx, how='all'):
    # Takes a dataframe to aggregate boolean data for and the columns to aggregate on
    df = df.copy()
    # Filtering down just to the categorical values
    df['dtype'] = df['value'].apply(dp.get_data_type)
    df = df.loc[df['dtype']=='str']
    # Converting value column numeric indexes related to unique states in the value column
    cat2idx = dict(map(reversed,pd.DataFrame(df['value'].unique()).to_dict()[0].items()))
    df['value'] = df['value'].apply(lambda x: cat2idx[x])
    # Converting indexes to group by to the column names
    group_names = df.columns[col_idx].values.tolist()
    group_names.extend(['hour']) # First aggregation is by hour
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
        temp_df = temp_df.groupby(group_names).agg({'value':['sum','count']},axis=1)
        if isFirst==True:
                return_df = temp_df
                isFirst = False
        else:
            return_df = return_df.append(temp_df)
    group_names = group_names[:-1] # Drop the added hours value to 
    return_df.columns = return_df.columns.droplevel()
    if how=='all':
        return_df = return_df.groupby(group_names).agg({'sum':['mean','std','max','min'], 'count':'sum'},axis=1) # Sum b/c the hours aggregation did the counts
        return_df.columns = return_df.columns.droplevel()
        return_df= return_df.rename(columns={'sum':'count'})
    else:
        return_df = return_df.groupby(group_names).agg({'sum':how,'count':'sum'},axis=1) # Sum b/c the hours aggregation did the counts
        if type(how)==list:
            return_df.columns = return_df.columns.droplevel()
            return_df= return_df.rename(columns={'sum':'count'})
        else:
            return_df= return_df.rename(columns={'sum':how})
    return return_df.fillna(0)

# Testing agg_bool_by_col() function
test_cat_agg = agg_cat_by_col(data, [1,2,3,4,5,8], ['mean','std'])

# Function to fun all three aggregation types covered by the above functions and output data columnwise for input into other functions (ex. feature selection, and scaling)
def agg_all(df, col_idx, how='all', last_idx_to_col=True):
    # Takes a dataframe to aggregate, columns to aggregate on, and how to aggregate the numeric values (bool and cat only have one type of aggregation), and if the last index should become columns (default=True)
    # Aggregating each datatype
    num_agg = agg_numeric_by_col(df, col_idx, how=how)
    bool_agg = agg_bool_by_col(df, col_idx, how=how)
    cat_agg = agg_cat_by_col(df, col_idx, how=how)
    # Combining all aggregations into one dataframe
    all_agg = num_agg.append(bool_agg)
    all_agg = all_agg.append(cat_agg)
    agg_cols = all_agg.columns.tolist()
    if last_idx_to_col == True:
        groups = all_agg.reset_index().iloc[:,-len(agg_cols)-1].unique()
        new_df = all_agg.reset_index().drop(all_agg.reset_index().columns[-len(agg_cols)-1:].tolist(), axis=1).drop_duplicates()
        onList = new_df.columns.tolist()
        is_first = True
        for group in groups:
            cur_df = all_agg.reset_index()[all_agg.reset_index().iloc[:,-len(agg_cols)-1]==group].drop(all_agg.reset_index().columns[-len(agg_cols)-1], axis=1)
            new_df = pd.merge(new_df, cur_df, on=onList, how='left')
            col_names = new_df.columns.tolist()
            col_names[-len(agg_cols):]=[i+"_"+str(group) for i in col_names[-len(agg_cols):]]
            new_df.columns = col_names
            new_df = new_df.fillna(0)
            if is_first:
                is_first = False
                first_group = group
            else:
                new_df['count_'+str(first_group)] += new_df['count_'+str(group)]
                new_df = new_df.drop(labels='count_'+str(group), axis=1)
        new_df = new_df.rename(columns={'count_'+str(first_group):'count'})
        all_agg = new_df.fillna(0)
    else:
        all_agg = all_agg.reset_index().fillna(0)
    return all_agg

# Testing the agg_all() function when the last index is split to columns (days)
test_all_agg = agg_all(data, [1,2,3,4,5,8], how='all')
# Testing the agg_all() function when the last index isn't split to columns
test_all_agg2 = agg_all(data, [1,2,3,4,5,8], how='std', last_idx_to_col=False)
# Testing the agg_all() function when the last index is split to columns (hours)
test_all_agg3 = agg_all(data, [1,2,3,4,5,10], how='median')
# Testing the agg_all() function when the last index is split to columns (groupRef)
test_all_agg4 = agg_all(data, [1,3,4,5,2], how='mean') # NOTE: This test was just to show that it can be done, it won't actually be of any use

##### ~~~ Setting up a test aggregation data for Eva ~~~ #####
# for_eva = agg_all(data, [1,2,3,4,5,6,8], how='all', last_idx_to_col=False)
# for_eva = for_eva.rename(columns={'count':'sensorrate'})
# for_eva.to_csv("sample_agg_data.csv")
##### ~~~ Setting up a test aggregation data for Eva ~~~ #####

df = data.copy()
col_idx = [1,2,3,4,5,10] # Columns from the original input dataframe (before aggregations)
append_data1 = split_datetime(pd.read_csv('test_data/2020-05-01.csv'))
append_data2 = split_datetime(pd.read_csv('test_data/2020-03-16.csv'))

test_app1 = agg_all(append_data1, col_idx, how='all')
test_app2 = agg_all(append_data2, col_idx, how='all')

test_app_check = test_app1.copy()
#test_set1 = set(test_all_agg.columns.tolist())
#print(test_set1)
#test_set2 = set(test_all_agg2.columns.tolist())
#print(test_set2)
#common_cols = list(test_set1 & test_set2)
#print(common_cols)


def append_agg(df1, df2, col_idx, last_idx_to_col=True):
    if last_idx_to_col==True:
        col_idx = col_idx[:-1]
    group_names = df.columns[col_idx].values.tolist()
    cols = list(set(df1.columns.tolist())-set(group_names))
    temp_df = pd.merge(df1,df2, how='outer', on=group_names, suffixes=['_1','_2'])
    for col in cols:
        if col=='count':
            temp_df['count'] = temp_df[col+'_1'] + temp_df[col+'_2']
        else:
            temp_df[col] = (temp_df[col+'_1']*temp_df['count_1']+temp_df[col+'_2']*temp_df['count_2'])/(temp_df['count_1']+temp_df['count_2'])
    dropList = [col+"_1" for col in cols]
    dropList.extend([col+"_2" for col in cols])
    temp_df = temp_df.drop(dropList, axis=1)
    temp_df = temp_df.fillna(0)
    return temp_df

test_append_agg = append_agg(test_app1, test_app2, [1,2,3,4,5,10])

test_app3 = agg_all(append_data1, col_idx, how='all', last_idx_to_col=False)
test_app4 = agg_all(append_data2, col_idx, how='all', last_idx_to_col=False)
test_append_agg2 = append_agg(test_app3, test_app4, [1,2,3,4,5,10], last_idx_to_col=False)

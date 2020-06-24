#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:06:05 2020

@author: connor
"""
# Library Imports
import pandas as pd
pd.options.mode.chained_assignment = None # Disabling chained_assignment warning as it is not a concern here
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning) # Disabling the lexsort performance warning as it has minimal relevance at this point

# Function imports from other files
import data_preparation as dp

def split_datetime(df):
    """Function to create and populate columns for the date, month, and hour value from the timestamp of each observation

    Args:
        df (pandas.DataFrame): dataframe containing a 'datetime' column
        
    Returns:
        pandas.DataFrame: input dataframe with 'date', 'month', 'hour' columns appended

    """
    df = df.copy()
    # Extracting the date and hour of the day from the timestamp column
    df.loc[:,'datetime'] = pd.to_datetime(df.loc[:,'datetime'])
    df.loc[:,'date'] = df.loc[:,'datetime'].dt.date.astype(str)
    df.loc[:,'month'] = df.loc[:,'datetime'].dt.month.astype(str)
    df.loc[:,'hour'] = df.loc[:,'datetime'].dt.hour.astype(str) # Want as str so that it doesn't get aggregated when not aggregating on it
    return df

def agg_numeric_by_col(df, col_idx, how='all'):
    """Function to aggregate numeric data in user specified column using specified aggregation function(s).
       
       The user can specify how='all' to get 'mean', 'std', 'max', 'min', and 'count', or they can specify any function
       that is accepted by the .agg function, this will return the provided string and the count. Finally the user 
       can pass in a list of functions that are accepted by the .agg function to get the specified aggregations and the
       count of observations. The function filters out any non-numeric values in the "value" column.

    Args:
        df (pandas.DataFrame): dataframe containing at least a 'value' column
        col_idx (list): one or more column indicies to group by when aggregating
        how (str or list): 'mean', 'std', 'max', 'min', 'all', (or any function that can be passed into the .agg function)
        
    Returns:
        pandas.DataFrame: Dataframe with col_idx columns as the indexesthe appropriate aggregation(s) as columns,
                          and the count of observations as a column
    """
    df = df.copy()
    # Filtering down just to the numeric values
    df.loc[:,'dtype'] = df.loc[:,'value'].apply(dp.get_data_type)
    df = df.loc[df.loc[:,'dtype']=='num']
    # Converting value column to float
    df.loc[:,'value'] = df.loc[:,'value'].astype(float)
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

def agg_bool_by_col(df, col_idx, how='all'):
    """Function to aggregate the count of boolean value changes the in user specified column(s) using specified aggregation function(s).
       
       The user can specify how='all' to get 'mean', 'std', 'max', 'min', and 'count', or they can specify any function
       that is accepted by the .agg function, this will return the provided string and the count. Finally the user 
       can pass in a list of functions that are accepted by the .agg function to get the specified aggregations and the
       count of observations. The function filters out any non-boolean values in the "value" column.

    Args:
        df (pandas.DataFrame): dataframe containing at least a 'value' column
        col_idx (list): one or more column indicies to group by when aggregating (The "hour" column must always be included for aggregation to work properly)
        how (str or list): 'mean', 'std', 'max', 'min', 'all', (or any function that can be passed into the .agg function)
        
    Returns:
        pandas.DataFrame: Dataframe with col_idx columns as the indexesthe appropriate aggregation(s) as columns,
                          and the count of observations as a column
    """
    df = df.copy()
    # Filtering down just to the boolean values
    df.loc[:,'dtype'] = df.loc[:,'value'].apply(dp.get_data_type)
    df = df.loc[df.loc[:,'dtype']=='bool']
    # Converting value column to 0 (False) and 1 (True)
    df.loc[:,'value'] = df.loc[:,'value'].apply(lambda x: 1 if x=='True' else 0)
    # Converting indexes to group by to the column names
    group_names = df.columns[col_idx].values.tolist()
    group_names.extend(['hour']) # First aggregation is by hour
    # Sorting observations by their timestamp
    df = df.sort_values(by='datetime')
    # Creating column for easy aggregation (always aggregate by hour first so do hour first)
    df.loc[:, 'groups'] = ''
    for idx in col_idx:
            df.loc[:,'groups'] += df.iloc[:,idx].astype(str)
    # Adding hr_idx to the end of groups for initial grouping on hours
    isFirst = True
    groupList = df.loc[:,'groups'].unique()
    # For loop to identify the number of changes for each group individually
    for group in groupList:
        temp_df = df.loc[df.loc[:,'groups']==group]
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

def agg_cat_by_col(df, col_idx, how='all'):
    """Function to aggregate the count of categorical value changes the in user specified column(s) using specified aggregation function(s).
       
       The user can specify how='all' to get 'mean', 'std', 'max', 'min', and 'count', or they can specify any function
       that is accepted by the .agg function, this will return the provided string and the count. Finally the user 
       can pass in a list of functions that are accepted by the .agg function to get the specified aggregations and the
       count of observations. The function filters out any non-boolean values in the "value" column.

    Args:
        df (pandas.DataFrame): dataframe containing at least a 'value' column
        col_idx (list): one or more column indicies to group by when aggregating (The "hour" column must always be included for aggregation to work properly)
        how (str or list): 'mean', 'std', 'max', 'min', 'all', (or any function that can be passed into the .agg function)
        
    Returns:
        pandas.DataFrame: Dataframe with col_idx columns as the indexes, the appropriate aggregation(s) as columns,
                          and the count of observations as a column
    """
    df = df.copy()
    # Filtering down just to the categorical values
    df.loc[:,'dtype'] = df.loc[:,'value'].apply(dp.get_data_type)
    df = df.loc[df.loc[:,'dtype']=='str']
    # Converting value column numeric indexes related to unique states in the value column
    cat2idx = dict(map(reversed,pd.DataFrame(df.loc[:,'value'].unique()).to_dict()[0].items()))
    df.loc[:,'value'] = df.loc[:,'value'].apply(lambda x: cat2idx[x])
    # Converting indexes to group by to the column names
    group_names = df.columns[col_idx].values.tolist()
    group_names.extend(['hour']) # First aggregation is by hour
    # Sorting observations by their timestamp
    df = df.sort_values(by='datetime')
    # Creating column for easy aggregation
    df.loc[:, 'groups'] = ''
    for idx in col_idx:
            df.loc[:,'groups'] += df.iloc[:,idx].astype(str)
    isFirst = True
    groupList = df.loc[:,'groups'].unique()
    # For loop to identify the number of changes for each group individually
    for group in groupList:
        temp_df = df.loc[df.loc[:,'groups']==group]
        temp_df.loc[:,'value'] = temp_df.loc[:,'value'].diff().abs()
        temp_df = temp_df.drop(labels='groups', axis=1)
        temp_df.loc[:,'value'] = temp_df.loc[:,'value'].fillna(0)
        temp_df.loc[:,'value'] = temp_df.loc[:,'value'].apply(lambda x: 1 if x>0 else 0)
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

def combine_mixed_agg(df, struct_df, col_idx):
    """Function to aggregate any duplicate observations within a dataframe, this function is required in the case that there are any mixed datatypes between 
        aggregation groups (ex. if an aggregation group has continuous and categorical data these would be stored as two aggregations and need to be aggregated further down to one observation)

    Args:
        df (pandas.DataFrame): dataframe to check for, and aggregate (if applicable) duplicate observations
        struct_df (pandas.DataFrame): dataframe with the structure of the original data (can just be the head from the original dataframe)
        col_idx (list): one or more column indicies corresponding to the struct_df indices of the column being grouped on
    Returns:
        pandas.DataFrame: Dataframe with any duplicate observations aggregated down to one observation
    """
    df = df.copy()
    group_names = struct_df.columns[col_idx].values.tolist()
    dupes = df.duplicated(subset = group_names, keep=False)
    dupes = df.loc[dupes]
    if len(dupes) == 0:
        return df
    else:
        df = df.drop_duplicates(subset=group_names, keep=False)
        dupes.loc[:,'id'] = ''
        for group in group_names:
            dupes.loc[:,'id'] += dupes.loc[:,group].astype(str)
        dupe_list = dupes['id'].unique()
        agg_types = list(set(df.columns.tolist())-set(group_names))
        for dupe in dupe_list:
            cur_dupe = dupes.loc[dupes.loc[:,'id']==dupe]
            #cur_dupe = pd.DataFrame(cur_dupe).transpose()
            dupe_agg = dupes.loc[dupes['id']==dupe].iloc[0,:]
            dupe_agg = pd.DataFrame(dupe_agg).transpose()
            dupe_agg.loc[:,'count'] = cur_dupe.loc[:,'count'].sum()
            for agg in agg_types:
                if agg == 'max':
                    dupe_agg.loc[:,'max'] = cur_dupe.loc[:,'max'].max()
                elif agg == 'min':
                    dupe_agg.loc[:,'min'] = cur_dupe.loc[:,'min'].min()
                elif agg == 'mean' or agg == 'std':
                    dupe_agg.loc[:,agg] = sum(cur_dupe.loc[:,agg]*cur_dupe.loc[:,'count'])/sum(cur_dupe.loc[:,'count'])
            dupe_agg = dupe_agg.drop('id', axis=1)
            df = df.append(dupe_agg)
        return df
            
    #drop_cols = list(set(nc_data.columns.tolist())-set(test_groups))

def agg_all(df, col_idx, how='all', last_idx_to_col=True):
    """Function to run all three aggregation types covered by the above functions and output data columnwise for input into other functions (ex. feature selection, and scaling)
       
       The user can specify how='all' to get 'mean', 'std', 'max', 'min', and 'count', or they can specify any function
       that is accepted by the .agg function, this will return the provided string and the count. Finally the user 
       can pass in a list of functions that are accepted by the .agg function to get the specified aggregations and the
       count of observations. The function filters out any non-numeric values in the "value" column. The user can also specify
       if the last index of the col_idx input should be made into columns or not.

    Args:
        df (pandas.DataFrame): dataframe containing at least a 'value' column
        col_idx (list): one or more column indicies to group by when aggregating
        how (str or list): 'mean', 'std', 'max', 'min', 'all', (or any function that can be passed into the .agg function)
        last_idx_to_col (bool): A boolean value indicating if the last index in the col_idx list should be made into columns in the output (True means yes and is the default)
        
    Returns:
        pandas.DataFrame: Dataframe with col_idx columns (minus the last index in the list if last_idx_to_col=True)
                          as the indexes, the appropriate aggregation(s) (grouped by the unique values of the column 
                          of the last value in col_idx if last_idx_to_col=True) as columns, and the count of observations
                          as a column
    """
    first_group=''
    # Aggregating each datatype
    num_agg = agg_numeric_by_col(df, col_idx, how=how)
    bool_agg = agg_bool_by_col(df, col_idx, how=how)
    cat_agg = agg_cat_by_col(df, col_idx, how=how)
    # Combining all aggregations into one dataframe
    all_agg = num_agg.append(bool_agg)
    all_agg = all_agg.append(cat_agg)
    agg_cols = all_agg.columns.tolist()
    if last_idx_to_col:
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
            new_df = new_df.drop_duplicates(subset = onList) ##### Added
            if is_first:
                is_first = False
                first_group = group
            else:
                new_df.loc[:,'count_'+str(first_group)] += new_df.loc[:,'count_'+str(group)]
                new_df = new_df.drop(labels='count_'+str(group), axis=1)
        new_df = new_df.rename(columns={'count_'+str(first_group):'count'})
        all_agg = new_df.fillna(0)
        all_agg = combine_mixed_agg(all_agg, df.head(1), col_idx[:-1])
    else:
        all_agg = all_agg.reset_index().fillna(0)
        all_agg = combine_mixed_agg(all_agg, df.head(1), col_idx)
    return all_agg

def append_agg(df1, df2, struct_df, col_idx, last_idx_to_col=True):
    """ Function to combine two previously aggregated dataframes with weighted averages for aggregations and total count for counts
       
       The user can pass in two aggregation dataframes (prefereably outputs from the agg_all function) in order
       to combine them. This function is intended for sequentially aggregating data in small portions since aggregating
       large datasets is infeasible due to limitations on RAM and computation power. NOTE: Both dataframes must be in the
       same format.

    Args:
        df1 (pandas.DataFrame): dataframe containing the first of two dataframes to aggregate
        df2 (pandas.DataFrame): dataframe containing the second of two dataframes to aggregate
        struct_df (pandas.DataFrame): the original dataframe used to generate df1 or df2 (just needs the structure so could just pass the head)
        col_idx (list): the column indicies to group by when aggregating
        how (str or list): 'mean', 'std', 'max', 'min', 'all', (or any function that can be passed into the .agg function)
        last_idx_to_col (bool): A boolean value indicating if the last index in the col_idx list should be made into columns in the output (True means yes and is the default)
        
    Returns:
        pandas.DataFrame: Dataframe with col_idx columns (minus the last index in the list if last_idx_to_col=True)
                          as the indexes, the appropriate aggregation(s) (grouped by the unique values of the column 
                          of the last value in col_idx if last_idx_to_col=True) as columns, and the count of observations
                          as a column
    """
    if last_idx_to_col==True:
        col_idx = col_idx[:-1]
    group_names = struct_df.columns[col_idx].values.tolist()
    #if 'hour' not in group_names:
    #    group_names.append('hour')
    cols = list(set(df1.columns.tolist())-set(group_names))
    temp_df = pd.merge(df1,df2, how='outer', on=group_names, suffixes=['_1','_2'])
    temp_df = temp_df.fillna(0)
    for col in cols:
        if col=='count':
            temp_df.loc[:,'count'] = temp_df.loc[:,col+'_1'] + temp_df.loc[:,col+'_2']
        else:
            try:
                temp_df.loc[:,col] = (temp_df.loc[:,col+'_1']*temp_df.loc[:,'count_1']+temp_df.loc[:,col+'_2']*temp_df.loc[:,'count_2'])/(temp_df.loc[:,'count_1']+temp_df.loc[:,'count_2'])
            except:
                if col+"_1" in temp_df.columns.tolist():
                    temp_df.loc[:,col] = temp_df.loc[:,col+'_1']
                    temp_df.loc[:,col+'_2'] = 0
                elif col+"_2" in temp_df.columns.tolist():
                    temp_df.loc[:,col] = temp_df.loc[:,col+'_2']
                    temp_df.loc[:,col+'_1'] = 0
                else:
                    temp_df.loc[:,col] = 0
                    temp_df.loc[:,col+'_1'] = 0
                    temp_df.loc[:,col+'_2'] = 0
    dropList = [col+"_1" for col in cols]
    dropList.extend([col+"_2" for col in cols])
    temp_df = temp_df.drop(dropList, axis=1)
    temp_df = temp_df.fillna(0)
    return temp_df
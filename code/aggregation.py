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



data['datetime'] = pd.to_datetime(data['datetime'])
data['date'] = data['datetime'].dt.date
data['hour'] = data['datetime'].dt.hour.astype(str) # Want as str so that it doesn't get aggregated when not aggregating on it

def agg_numeric_by_col(df, col_idx):
    # Takes a dataframe to aggregate numeric data for and the columns to aggregate on
    df = df.copy()
    # Filtering down just to the numeric values
    df['dtype'] = df['value'].apply(dp.get_data_type)
    df = data.loc[data['dtype']=='num']
    # Converting value column to float
    df['value'] = df['value'].astype(float)
    # Get column names to aggregate by
    groupNames = df.columns[col_idx].values.tolist()
    # Aggregate and return values
    return df.groupby(groupNames).mean()

# Testing aggregating on numeric function using unique ID columns and hour of the day
agg_numeric_by_col(data, [1,2,3,4,5,9])
# Testing aggregating on numeric function using unique ID columns and date
agg_numeric_by_col(data, [1,2,3,4,5,8])
# Testing aggregating on numeric function using unit column and hour of the day
agg_numeric_by_col(data, [6,9])
# Testing aggregating on numeric function using unit columns and date
agg_numeric_by_col(data, [6,8])

# For easily viewing the data (Delete when done)
test = data.sample(20)
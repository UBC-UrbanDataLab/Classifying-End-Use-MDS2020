# Only the sensor update rate aggregation
# Will add this into the aggregation.py file once tested

import pandas as pd
import numpy as np



'''
agg_update_rate
arguments:
    dataframe = pandas dataframe with total number of updates each day for each sensor over some date range
returns:
    A dataframe with sensor id tags and columns for average number of updates per day, variance in number of updates per day

'''

def agg_up_rate(dataframe):
    try:
        grouped = dataframe.groupby(["equipRef","groupRef","navName","typeRef","unit"])
        num_updates_df = grouped.agg([np.mean,np.median,np.std,np.max]) #no reason to include min. Will be 0 due to server outage days
        num_updates_df.reset_index(inplace=True)
        return(num_updates_df)
    except:
        print("Error in agg_up_rate()")
        return(0)

# Test the function

test_df = pd.read_csv("test_data/numSensorUpdatesApr01ToMay20.csv")

results = agg_up_rate(test_df)

print(results.head(20))


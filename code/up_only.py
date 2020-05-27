# Only the sensor update rate aggregation
# Will add this into the aggregation.py file once tested

import pandas as pd
import numpy as np





def agg_up_rate(dataframe):
    """Function for calculating aggregated daily sensor update rates

    Args:
        dataframe (pandas.dataFrame): dataframe containing sensor id tags, date, and number of updates for the day
        
    Returns
        Pandas.dataFrame: dataframe with sensor id tags and columns for mean number of updates per day, variance in number of updates per day, and max number of updates per day

    """

    try:
        grouped = dataframe.groupby(["equipRef","groupRef","navName","typeRef","unit"])
        num_updates_df = grouped.agg([np.mean,np.median,np.std,np.max]) #no reason to include min. Will be 0 due to server outage days
        num_updates_df.reset_index(inplace=True)
        return(num_updates_df)
    except:
        #print("Error in agg_up_rate()")
        return(0)

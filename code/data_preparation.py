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

def fix_units_incons(nav, equip, val, u, typeref):
    """Function to analyze a set of values for and correct the unit of measurement if needed.
        The rules used to make the decision are based on errors found in actual data.

    Args:
        nav (string): navName value
        equip (string): equipRef value
        val (string): value field
        typeref (string): typeRef value
    Returns:
        The correct unit of measurement for that entry.
    """
   # rows 2-4 on Connor's csv
    #print("executing fix_units_incons")

    if nav.find('ALRM')!=-1:
        try:
            val = float(val)
        except:
            val = float(bool(val))
        if (val==1 or val==0):
            return "omit"
    # rows 5-7, 40-42, 51-53 on Connor's csv
    elif nav.find('Enable Cmd')!=-1 or nav.find('Run Cmd')!=-1:
        try:
            val = float(val)
        except:
            val = float(bool(val))
        if (val==1 or val==0):
            return "omit"
    # rows 8-13 on Connor's csv
    elif nav.find('Outside Air Damper')!=-1:
        try:
            val = float(val)
        except:
            val = float(bool(val))
        if (val==1 or val==0):
            return "omit"
    # rows 14-15 on Connor's csv
    elif (nav=='Hot Water Flow') and (u=='°C'):
        return "gal/min"
    # rows 16-19 on Connor's csv
    elif (nav=='ESB_CHIL1_PCE01_CHWFLW' or nav=='ESB_CHIL2_PCE02_CHWFLW') and (u=='°C'):
        return "gal/min"
    # rows 22-23, 64-65, 74-97, 104-115 on Connor's csv
    elif (val=='True'):
        return "omit"
    # rows 24-31 on Connor's csv
    elif (nav=='Valve Feedback') and (u=='_'):
        return '%'
    # rows 32-33, 54-55 on Connor's csv
    elif nav=='Exhaust Air Duct Static Pressure' and (u=='°C' or u=='ft/min'):
        return "Pa"
    # rows 43-33 on Connor's csv
    elif nav.find('Speed')!=-1 and (u=='°C'):
        return "%"
    # rows 56-63 on Connor's csv
    elif nav.find('JCL-NAE43/FCB-01.FEC-50.BLDG')!=-1 and (u=='Pa'):
        return "°C"
    # rows 66-67, 70-73 on Connor's csv
    elif (nav=='Heating Valve Cmd') and (u=='_' or u=='A'):
        return "%"
    # rows 68-69 on Connor's csv
    elif nav.find('Discharge Air Damper')!=-1 and (u=='_'):
        return "%"
    # rows 98-103, 124-135 on Connor's csv
    elif nav.find('Flow Alarm')!=-1 and (u=='%' or u=='_' or u=='L/s'):
        return "omit"
    # rows 116-123 on Connor's csv
    elif nav=='Exhaust Air Flow' and (u=='°C'):
        return "L/s"
    # rows 136-171, 184-291 on Connor's csv
    elif nav.find('Water Temp')!=-1 and (u=='%'):
        return "°C"
    # rows 172-183 on Connor's csv
    elif nav.find('ESB_TMX')!=-1 and equip.find('Thermanex Header')!=-1 and u=='%':
        return "°C"
    # rows 220-221 on Connor's csv
    elif nav.find('Outside Air Temp')!=-1 and equip.find('Thermanex Header')!=-1 and u=='%':
        return "°C"
    ##### unsure rows ######
    # rows 20-21 on Connor's csv
    elif nav.find('JCL-NAE29/BACnet-IP.FOL-MHP4.Analog Values.AV-1002')!=-1 and u=='_':
        return "°C"
    # rows 34-50 on Connor's csv
    elif nav.find('ISOD')!=-1 and equip.find('LEF-4 EF-4')!=-1 and (u=='_' or u=='°C'):
        return "omit"
    # changing all '_' of kWh typeRefs  to kWh from Alex's training data
    elif u.find('_')!=-1 and typeref.find('kWh')!=-1:
        return "kWh"
    # changing all '_' units to 'omit' to standardize unknowns from Alex's training data
    elif u.find('_')!=-1:
        return "omit"
    else:
        return u

def correct_df_units(df):
    """Function to correct units of measurement column in dataframe

    Args:
        df (pandas Dataframe): dataframe containing the columns uniqueId, navName, equipRef, value, and unit
    Returns:
        inputted dataframe with values in units column corrected
    """
    df_with_update = df.copy()
    try:
        # grabbing sensor ids with problematic uom
        problematic_sensors=df_with_update[['uniqueId','unit']].drop_duplicates()[df_with_update[['uniqueId','unit']].drop_duplicates()['uniqueId'].duplicated(keep=False)].uniqueId.unique()
        # creating a boolean column if sensor_id problematic
        sensor_check=df_with_update.uniqueId.isin(problematic_sensors)
        # inserting new boolean column
        df_with_update.loc[:,'prob_check']=pd.Series(sensor_check, index=df_with_update.index)
        # separating dataframe with problematic sensors
        newdf1=df_with_update[df_with_update['prob_check']==True]
        mod_unit=newdf1.apply(lambda x: fix_units_incons(x.navName, x.equipRef, x.value, x.unit, x.typeRef), axis=1)
        newdf1.loc[:,'unit']=mod_unit
        # separating dataframe without problematic sensors
        newdf2=df_with_update[df_with_update['prob_check']==False]
        # combining the two dataframes
        final_df=pd.concat([newdf1, newdf2])
        # drop boolean column
        final_df.drop(['prob_check'], axis=1, inplace=True)
        # return original df with fixed uom
        return final_df
    except ValueError as e:
        print(e)
        return None
    except:
        print("Failed to execute correct_df_units()")
        return None

def equip_label(equip):
    """Function to group equipRef categorical levels into smaller ones
    Args:
        equip (str): equipRef categorical level

    Returns:
        (str): a smaller grouped categorical level
    """
    if equip.find('Cooling')!=-1 or equip.find('CT')!=-1:
        return 'Cooling'
    elif equip.find('AHU')!=-1:
        return 'Air_Equip'
    elif equip.find('Windows')!=-1:
        return 'Window'
    elif equip.find('VAV')!=-1:
        return 'VAV'
    elif equip.find('Heating')!=-1:
        return 'Heating'
    elif equip.find('RAD')!=-1:
        return 'RAD'
    elif equip.find('EF')!=-1:
        return 'Air_Equip'
    elif equip.find('LEF')!=-1:
        return 'LEF'
    elif equip.find('FF')!=-1:
        return 'Fan_Equip'
    elif equip.find('FM')!=-1:
        return 'Air_Equip'
    elif equip.find('EAV')!=-1:
        return 'EAV'
    elif equip.find('PA')!=-1:
        return 'OPC(TV)'
    elif equip.find('PB')!=-1:
        return 'OPC(TV)'
    elif equip.find('FC')!=-1:
        return 'Fan_Equip'
    elif equip.find('CRAH')!=-1:
        return 'Air_Equip'
    elif equip.find('LEED')!=-1:
        return 'LEED'
    elif equip.find('Zone')!=-1:
        return 'Humidity'
    elif equip.find('WM')!=-1:
        return 'Water'
    elif equip.find('Gas')!=-1:
        return 'Gas'
    elif equip.find('DCB')!=-1:
        return 'Power'
    elif equip.find('DCA')!=-1:
        return 'Power'
    else:
        return "NEED TO LABEL"

def nav_label(nav):
    """Function to group equipRef categorical levels into smaller ones
    Args:
        equip (str): equipRef categorical level

    Returns:
        (str): a smaller grouped categorical level
    """
    if nav.lower().find('alarm')!=-1:
        return 'Alarm'
    elif nav.lower().find('temp')!=-1 or nav.lower().find('lwt')!=-1 or nav.lower().find('ewt')!=-1 or nav.lower().find('humidity')!=-1:
        #leaving and entering water temperature # humidity sensors measures moisture&air temps
        return 'Temp'
    elif nav.lower().find('water')!=-1 or nav.lower().find('_cw')!=-1 or nav.lower().find('chw')!=-1 or nav.find('SB1_2_FWT_T')!=-1 or nav.lower().find('lwco')!=-1:
        # I think FW = Feed Water # CW = Condenser Water # CHW = Chilled Water metrics # LWCO = low water cut off
        return 'Water'
    elif nav.lower().find('air')!=-1 or nav.lower().find('ach')!=-1 or nav.lower().find('ahu')!=-1 or nav.lower().find('inlet')!=-1:
        # AHU = Air Handling Unit # ACH = Air Changes per Hour # Inlet Air Temperature sensor
        return 'Air'
    elif nav.lower().find('press')!=-1 or nav.lower().find('_dp')!=-1: # DP = differential pressure
        return 'Pressure'
    elif nav.lower().find('heat')!=-1 or nav.lower().find('hrv')!=-1 or nav.lower().find('_rh')!=-1: # HRV = heat recovery ventilator # RH = Reheat
        return 'Heat'
    elif nav.lower().find('fire_rate')!=-1 or nav.lower().find('firing_rate')!=-1:
        return 'Fire Rate'
    elif nav.lower().find('power')!=-1 or nav.lower().find('voltage')!=-1 or nav.lower().find('vfd')!=-1: # VFD = variable frequency drive
        return 'Power'
    elif nav.lower().find('energy')!=-1 or nav.lower().find('curr')!=-1 or nav.lower().find('btu')!=-1 or nav.find('kW')!=-1: # eletrical current
        return 'Energy'
    elif nav.lower().find('fan')!=-1 or nav.lower().find('fcu')!=-1 or (nav.lower().find('ef')!=-1 and  nav.lower().find('efficiency')==-1): # FCU = fan coil unit # EF = exhaust fan
        return 'Fan'
    elif nav.find('Instant_Power')!=-1:
        return 'Instant_Power'
    elif nav.lower().find('open_percent')!=-1 or nav.lower().find('occupancy')!=-1:
        return 'Occupancy'
    elif nav.lower().find('feedback')!=-1 or nav.lower().find('demand')!=-1: # demand controlled ventilation
        return 'Feedback'
    elif nav.find('CO2')!=-1:
        return 'CO2'
    elif nav.lower().find('power')!=-1:
        return 'Power'
    elif nav.lower().find('cool')!=-1 or nav.lower().find('_ct_')!=-1: # CT = cooling tower
        return 'Cooling'
    elif nav.lower().find('speed')!=-1:
        return 'Speed'
    elif nav.lower().find('pump')!=-1:
        return 'Pump'
    elif nav.lower().find('_tl')!=-1:
        return '_TL'
    elif nav.lower().find('_aflw')!=-1:
        return '_AFLW'
    elif nav.lower().find('_sp')!=-1:
        return '_SP/_SPT'
    elif nav.lower().find('cmd')!=-1:
        return 'Cmd'
    elif nav.lower().find('_day')!=-1:
        return '_DAY'
    elif nav.lower().find('_av')!=-1:
        return '_AV'
    elif nav.lower().find('_bms')!=-1:
        return '_BMS'
    elif nav.lower().find('status')!=-1:
        return '_Status'
    elif nav.lower().find('rwt')!=-1:
        return '_RWT'
    elif nav.lower().find('_open')!=-1:
        return '_OPEN'
    elif nav.lower().find('wifi')!=-1:
        return 'Wifi'
    elif nav.lower().find('operation')!=-1:
        return 'Operation'
    elif nav.lower().find('pres')!=-1:
        return '_PRES'
    elif nav.lower().find('_efficiency')!=-1:
        return '_EFFICIENCY'
    elif nav.lower().find('_flow')!=-1:
        return '_FLOW'
    elif nav.lower().find('_delay')!=-1:
        return '_DELAY'
    elif nav.lower().find('_clg')!=-1:
        return '_CLG'
    elif nav.lower().find('bs050')!=-1:
        return 'BS050'
    elif nav.lower().find('fdbk')!=-1:
        return '_FDBK'
    else:
        return "NEED TO LABEL"



def create_unique_id(df, metadata=False, indexes=['equipRef', 'groupRef', 'navName', 'siteRef', 'typeRef', 'bmsName']):
    """
    Function to add uniqueIds for sensors to a dataframe
    Args:
        df (dataframe): any pandas dataframe
        metadata (bool): False if data is not metadata
        indexes (list): list of field names
    Returns:
        df (dataframe): original dataframe with sensor uniqueId
    """
    if metadata==False:
        # concatenates the 5 fields
        uniqueid=df[indexes[0]].fillna('')+' '+df[indexes[1]].fillna('')+' '+df[indexes[2]].fillna('')+' '+df[indexes[3]].fillna('')+' '+df[indexes[4]].fillna('')
        # removes Pharmacy from uniqueId
        uniqueid=uniqueid.str.replace('Pharmacy ', '')
        # moves uniqueId to the front of df
        df.insert(0, 'uniqueId', uniqueid)
        return df
    elif metadata==True:
        # removes the database id from equipRef, groupRef, siteRef
        df[indexes[0]]=df[indexes[0]].str.extract('[^ ]* (.*)', expand=True)
        df[indexes[1]]=df[indexes[1]].str.extract('[^ ]* (.*)', expand=True)
        df[indexes[3]]=df[indexes[3]].str.extract('[^ ]* (.*)', expand=True)
        # concatenates the 5 fields
        uniqueid=df[indexes[0]].fillna('')+' '+df[indexes[1]].fillna('')+' '+df[indexes[2]].fillna('')+' '+df[indexes[3]].fillna('')+' '+df[indexes[5]].fillna('')
        # removes Pharmacy from uniqueId
        uniqueid=uniqueid.str.replace('Pharmacy ', '')
        # moves uniqueId to the front of df
        df.insert(0, 'uniqueId', uniqueid)
        return df
        
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

def query_db_ec(client, date, num_days=1, site='Pharmacy'):
    """Function to query the UBC_EWS database for the EC sensors for the user defined start date, 
       number of days (default=1), and site (default=Pharmacy)

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
        query = 'select * from UBC_EWS where siteRef=$siteRef and (unit=$unit1 or unit=$unit2 or navName=~/^(.*?(\bEnergy\b)[^$]*)$/ or typeRef=~/^(.*?(\bkWh\b)[^$]*)$/) and (time > $time1 and time < $time2)'
        where_params = {'siteRef':site,'unit1': 'kWh', 'unit2':'m³', 'time1':date+' '+time1, 'time2':date+' '+time2}
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
    
def query_db_nc(client, date, num_days=1, site='Pharmacy'):
    """Function to query the UBC_EWS database for the Non-Energy Consumption (NC) sensors 
       for the user defined start date, number of days (default=1), and site (default=Pharmacy)

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
        query = 'select * from UBC_EWS where siteRef=$siteRef and ((unit!=$unit1 and unit!=$unit2 and navName!~/^(.*?(\bEnergy\b)[^$]*)$/ and typeRef!~/^(.*?(\bkWh\b)[^$]*)$/) or groupRef=$groupRef )and (time > $time1 and time < $time2)'
        where_params = {'siteRef':site,'unit1': 'kWh', 'unit2':'m³', 'groupRef':'weatherRef','time1':date+' '+time1, 'time2':date+' '+time2}
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
        temp_df = pd.read_csv('test_data/'+filename)
        return temp_df
    except ValueError as e:
        print("ERROR: ", e)
        return None
    except OSError as e:
        print("ERROR: Unable to find or access file:", e)
        return None
    except:
        return None
    
def query_weather_csv(client, date, site):
    """Function to read the csv of saved data from the influxDB for the specified date. Requires csv files
    to already be saved in a weather_data subfolder. This is a temporary function to make testing faster while
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
        temp_df = pd.read_csv('weather_data/'+filename)
        return temp_df
    except ValueError as e:
        print("ERROR: ", e)
        return None
    except OSError as e:
        print("ERROR: Unable to find or access file:", e)
        return None
    except:
        return None

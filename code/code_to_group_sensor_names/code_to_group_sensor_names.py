#!/usr/bin/env python
# coding: utf-8

# ## LOADING INFLUXDB PHARMACY DATA

# In[6]:


######### PLEASE DOWNLOAD DATA FROM HERE 
# https://u.pcloud.link/publink/show?code=kZRQCKkZgc9dnKRyOOH0NVV4Dk9jRm2izqnk#folder=6009049826&tpl=publicfoldergrid 


############## CONNOR'S CSV FOR ALL OF JANUARY 2020 ###############
import os 
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##################### pulling csv files ##################### 
files1=glob.glob('/users/evanguyen/SkySpark_data/data_files/**.csv', recursive=True)

list1 = []

################  reading each csv filename into a dataframe into a list ############# 
for file_ in files1:
    df1 = pd.read_csv(file_,index_col=None, header=0)
    list1.append(df1)
      
        
##################### merging all the reviews files into one df ##################### 
for file_ in files1:
    df = pd.concat(list1)


# In[7]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# ## CREATING SMALLER GROUPS FOR EQUIPREF

# In[8]:


grouped=df.groupby('equipRef').count().sort_values(by='value', ascending=False)
grouped.reset_index(inplace=True)
def equip_label(equip):
    if equip.find('Cooling')!=-1:
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
grouped['equipNew']=grouped.equipRef.apply(lambda x: equip_label(x))
grouped.head(200)


# ## CREATING SMALLER GROUPS BASED ON NAVNAME

# In[9]:


grouped=df.groupby('navName').count().sort_values(by='value', ascending=False)
grouped.reset_index(inplace=True)

def nav_label(nav):
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
        return "LOWER FREQUENCY UNKNOWNS"

grouped['navNew']=grouped.navName.apply(lambda x: nav_label(x))
grouped.head(100)


#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
# loading NRCan classifications
training=pd.read_csv('/users/evanguyen/data-599-capstone-ubc-urban-data-lab/data/PharmacyEnergyConsumption-secondtry - PharmacyEnergyConsumption-secondtry.csv')
# making uniqueID 
training['siteRef']='Pharmacy'
training['uniqueId']=training['equipRef'].fillna('')+' '+training['groupRef'].fillna('')+' '+training['navName'].fillna('')+' '+training['siteRef'].fillna('')+' '+training['typeRef'].fillna('')
# need to rename column to use in fix_units_incons function below
training.rename(columns={'UBC_EWS.firstValue':'firstValue'}, inplace=True)


# In[4]:


################### CODE TO FIX INCONSISTENT DATA ####################
# creating a function to fix units 
def fix_units_incons(nav, equip, val, u, typeref):
    # rows 2-4 on Connor's csv
    if nav.find('ALRM')!=-1 and (float(val)==1 or float(val)==0):
        return "omit"
    # rows 5-7, 40-42, 51-53 on Connor's csv
    elif nav.find('Enable Cmd')!=-1 and (float(val)==1 or float(val)==0):
        return "omit"
    # rows 8-13 on Connor's csv
    elif nav.find('Outside Air Damper')!=-1 and (float(val)==1 or float(val)==0):
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
    elif nav.find('Water Temp') and (u=='%'):
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

mod_units=training.apply(lambda x: fix_units_incons(x.navName, x.equipRef, x.firstValue, x.unit, x.typeRef), axis=1)

# inserting new units as a new column
training.insert(6,"mod_units", mod_units)


# In[5]:


# overwrite unit column with fixed_uoms
training['unit']=training['mod_units']


# In[6]:


# drop unncessary columns in order to drop duplicate rows
training=training.drop(['Alex-Comments', 'UBC_EWS.numReadings', 'time','firstValue','UBC_EWS.lastValue'], axis=1)
# can change ? to 0 since uom fixed 
training['isGas']=training.isGas.apply(lambda x: '0' if x=='?' else x)
training=training.drop_duplicates()


# In[7]:


############### METADATA CLEANING ##############
metadata=pd.read_csv('/users/evanguyen/data-599-capstone-ubc-urban-data-lab/data/PharmacyQuery.csv')
##### Removing @UUID for now 
metadata['equipRef']=metadata['equipRef'].str.extract('[^ ]* (.*)', expand=True)
metadata['groupRef']=metadata['groupRef'].str.extract('[^ ]* (.*)', expand=True)
metadata['siteRef']=metadata['siteRef'].str.extract('[^ ]* (.*)', expand=True)
metadata['connRef']=metadata['connRef'].str.extract('[^ ]* (.*)', expand=True)

#### Making uniqueID 
metadata['uniqueId']=metadata['equipRef'].fillna('')+' '+metadata['groupRef'].fillna('')+' '+metadata['navName'].fillna('')+' '+metadata['siteRef'].fillna('')+' '+metadata['bmsName'].fillna('')
#### Dropping duplicate uniqueIDs based on most recent lastSynced
metadata=metadata.sort_values('lastSynced').drop_duplicates('uniqueId',keep='last')
### Choosing relevant fields
metadata=metadata[['uniqueId', 'connRef', 'kind', 'energy','power', 'sensor', 'unit', 'water']]
### Changing boolean to easily identify during encoding process
metadata['energy']=metadata['energy'].apply(lambda x: 'yes_energy' if x=='✓' else 'no_energy')
metadata['power']=metadata['power'].apply(lambda x: 'yes_power' if x=='✓' else 'no_power')
metadata['sensor']=metadata['sensor'].apply(lambda x: 'yes_sensor' if x=='✓' else 'no_sensor')
metadata['water']=metadata['water'].apply(lambda x: 'yes_water' if x=='✓' else 'no_water')
metadata['unit']=metadata['unit'].apply(lambda x: 'omit' if x=='_' else x)


# In[8]:


######## Removing the word Pharmacy from uniqueID
metadata['uniqueId']=metadata['uniqueId'].map(lambda x: x.replace('Pharmacy ', ''))
training['uniqueId']=training['uniqueId'].map(lambda x: x.replace('Pharmacy ', ''))


# In[9]:


merged_left=pd.merge(left=training, right=metadata, how='left', left_on='uniqueId', right_on='uniqueId')


# In[10]:


########## GROUPING EQUIPREF AND NAVNAME INTO SMALLER CATEGORICAL LEVELS #############
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
merged_left['equipNew']=merged_left.equipRef.apply(lambda x: equip_label(x))


# In[11]:


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

merged_left['navNew']=merged_left.navName.apply(lambda x: nav_label(x))


# In[12]:


###### renaming unit columns to reflect data source 
merged_left.rename(columns={'mod_units':'influxDB_units'}, inplace=True)
merged_left.rename(columns={'unit_y':'metadata_units'}, inplace=True)
##### selecting relevant fields 
merged_left=merged_left[['uniqueId','groupRef', 'influxDB_units', 'metadata_units', 'isGas', 'ALEX-NRCanLabelGuess', 'kind', 'energy','power', 'sensor', 'water', 'equipNew', 'navNew']]
merged_left.head()


# In[23]:


merged_left.to_csv("mergeddata.csv")


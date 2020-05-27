#!/usr/bin/env python
# coding: utf-8

# In[13]:


############ CONNOR'S CODE TO IDENTIFY INCONSISTENT DATA ############
# Data Import
import pandas as pd
df = pd.read_csv('all_sites_2020-04-01.csv')
df = pd.concat([df, pd.read_csv('all_sites_2020-04-02.csv')])
# Dropping weatherRef items (not part of this investigation)
df = df[df['groupRef']!='weatherRef']
# Creating a unique identifier
df['unique'] =  df['equipRef']+df['groupRef']+df['navName']+df['siteRef']+df['typeRef']
# Dropping Duplicate values once including units once not including units to see if there is a difference
no_dupes_w_unit = df.drop_duplicates(subset=['unique','unit'])
no_dupes_wo_unit = df.drop_duplicates(subset=['unique'])
# Comparing the number of observations after the two drops
print("# of observations when including units in the drop keys:" +str(len(no_dupes_w_unit)))
print("# of observations when omitting units from the drop keys:" +str(len(no_dupes_wo_unit)))
# Extracting the instruments with inconsistent units
units_incons = no_dupes_w_unit[no_dupes_w_unit.duplicated(subset=['unique'], keep=False)]
print("# of items in the duplicates list: "+str(len(units_incons)))
# Storing the list of instruments with inconsistent units in a csv
# units_incons.to_csv("units_incons.csv")


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

mod_units=units_incons.apply(lambda x: fix_units_incons(x.navName, x.equipRef, x.value, x.unit, x.typeRef), axis=1)

# inserting new units as a new column
units_incons.insert(6,"mod_units", mod_units)

# checking work
units_incons[units_incons.typeRef.str.find('JCL-NAE43/FCB-01.FEC-50.BLDG')!=-1]


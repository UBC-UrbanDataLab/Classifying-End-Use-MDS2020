#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd
import glob

###################### Connor's CSVs ##########################
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
    
###### Creating uniqueID 
df['uniqueId']=df['equipRef'].fillna('')+' '+df['groupRef'].fillna('')+' '+df['navName'].fillna('')+' '+df['siteRef'].fillna('')+' '+df['typeRef'].fillna('')


############### METADATA CLEANING ##############
metadata=pd.read_csv('PharmacyQuery.csv')
##### Removing @UUID for now 
metadata['equipRef']=metadata['equipRef'].str.extract('[^ ]* (.*)', expand=True)
metadata['groupRef']=metadata['groupRef'].str.extract('[^ ]* (.*)', expand=True)
metadata['siteRef']=metadata['siteRef'].str.extract('[^ ]* (.*)', expand=True)
metadata['connRef']=metadata['connRef'].str.extract('[^ ]* (.*)', expand=True)

#### Making uniqueID 
#### 'Pharmacy' in metadata equipRef, but not in Influx SkySpark
metadata.equipRef=metadata.equipRef.replace(regex=['Pharmacy'], value='').str.strip()
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


# In[93]:


######## Removing the word Pharmacy from uniqueID
metadata['uniqueId']=metadata['uniqueId'].map(lambda x: x.replace('Pharmacy ', ''))
df['uniqueId']=df['uniqueId'].map(lambda x: x.replace('Pharmacy ', ''))


# In[100]:


####### Left inner join 
merged_left=pd.merge(left=df, right=metadata, how='left', left_on='uniqueId', right_on='uniqueId')
merged_left.head()


# In[99]:


###### Reviewing number of unmerged uniqueIDs
number_of_unmerged=merged_left[merged_left.connRef.isnull()==True].uniqueId.nunique()
number_of_sensors_total=merged_left.uniqueId.nunique()

print("Number of unmerged sensors:", number_of_unmerged)
print("Number of total sensors:", number_of_sensors_total)
print("Percentage of unmerged sensors:", number_of_unmerged/number_of_sensors_total)


#####################
# select_md_fields.py
# 1) Outputs a csv with a list of every unique field in the Pharmacy building's metadata
# and how many buildings contain
# that field
# 2) Combines all exported building metadata CSV's into a single file that only contains
# the fields (columns) common across all of the buildings

import os
from pathlib import Path
import pandas as pd


#Expects all metadata to be in csv format stored in a folder named 'data'
#Load all csv's, storing them as a list of dataframes
#Also specifically load the pharmacy data csv in and store in its own dataframe

file_list = Path().glob("./data/*.csv")
df_list = []
for file in file_list:
    tempdf = pd.read_csv(file, low_memory=False)
    df_list.append(tempdf)
tempdf = None

pharm_df = pd.read_csv("./data/PharmacyQuery.csv", low_memory=False)

#Go through and count how many times a column appears across all the individual building csv files.
field_count = {}
for df in df_list:
    results = list(df.columns.values)
    for item in results:
        try:
            field_count[item] += 1
        except:
            field_count[item] = 1

print(len(df_list),"buildings processed.")
print(len(field_count),"unique columns found")
#Store a count of how many buildings were processed (useful for calculating percentages later)
field_count["TOTAL_BUILDINGS"]=len(df_list)

pharm_fields = list(pharm_df.columns.values)
phfield_count = {}
for key in field_count:
    if key in field_count:
        phfield_count[key]=field_count[key]

pharmCountdf = pd.DataFrame.from_dict(phfield_count, orient="index")
pharmCountdf.to_csv("ColumnOccuranceCount_PharmacyFieldsOnly.csv")

#Following code block can be uncommented to save a csv with all fields.
#I found this wasn't very useful given we are constrained to using the Pharmacy building
#   as our input, so there is no reason to care about fields that aren't contained in the Pharmac building
#   dataset.

#countdf = pd.DataFrame.from_dict(field_count, orient="index")
#print(countdf.head())
#countdf.to_csv("ColumnOccuranceCount_AllBuildings.csv")


########################
#2) Combine all of the dataframes, keeping only the fields common to all X% of them
# Note that the starting point is all fields in the Pharmacy building and it is only
# possible to select fewer fields than this - ie even if all buildings except Pharmacy
# have a field in common, it won't be selected.

#Get field list
n = int(0.95 * len(df_list)) #Select fields that at least 95% of the buildings have in common
print("Only selecting Pharmacy Building metadata fields common to",n,"out of",len(df_list),"buildings.")
fieldList = []
for key in phfield_count:
    if phfield_count[key]>= n:
        fieldList.append(key)
fieldList.remove("TOTAL_BUILDINGS") #Remove remanent from building the original phfield_count

#rebuild the dataframe list into one that only has the common fields
#Can't seem to get columns to update when using pd.dataframe.assign()...so I'm stuck with having a "view vs. copy" warning
for i in range(0,len(df_list),1):    
    df_list[i]=df_list[i][df_list[i].columns & fieldList]
    colsConverted = []
    for col in df_list[i].columns:
        if chr(10003) in str(df_list[i][col].values): #replace checkmark ascii symbols with 1            
            df_list[i][col]=df_list[i][col].apply(lambda x: 1 if str(x) == chr(10003) else 0)
            colsConverted.append(col)
            #dflist[i].astype({col:"int32"}) #This should work but isn't
#Concatenate all of the filtered dataframes into one master dataframe
masterdf = pd.concat(df_list)

print(len(fieldList),"fields selected based on selection of fields common to at least",n,"out of",len(df_list),"buildings")
print("Dimensions of resulting dataframe are:",masterdf.shape)

masterdf.to_csv("AllMetadataWithCommonFields.csv")
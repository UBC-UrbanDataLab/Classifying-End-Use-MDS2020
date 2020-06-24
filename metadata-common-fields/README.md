# Common Fields in "Metadata"

This folder contains code and data for exploring POINTS data exported from [EWS's SkySpark Database](https://energy.ubc.ca/energy-and-water-data/skyspark/). This was refered to as "metadata" in the UBCO MDS2020 Capstone Project. The main purpose of the code was to find common fields between the various buildings (not every building has the same fields!).

The code [select_md_fields.py](select_md_fields.py) is fully commented but the main purpose of it is to process all of the individual building's csv files stored in the data subfolder and find the fields that are common amongst them. As an initial filter, only the fields located in the Pharmacy Building's file are allowed (i.e. if it isn't in the Pharmacy Building it isn't even considered). This is then whittled down to fields that also exist in at least 95% of all the other buildings. 

Note that the 95% criteria is hard-coded in on line 68 of the python file. This is easily edited if you wish to view, for instance, only the fields that are common to all (100%) of the buildings:

`n = int(0.95 * len(df_list)) #Select fields that at least 95% of the buildings have in common`

Multiple csvs of results are outputted but the one of most interest may be "AllMetadataWithCommonFields.csv" as it shows all of the POINTS data across all buildings though for only the fields (columns) that are common to 95% of those buildings. 

&nbsp;  


***

A few notes on the buildings that were queried from SkySpark:

**All buildings queried between 2020-05-12 and 2020-05-13**

The following buildings exist in UDL's influxDB mirror of the skyspark database but do not exist in the skyspark database!

- Rusty Hat
- Totem CSNM

(They do not have any data for the last 30 days though...so may not matter)


The following buildings have no data (query reports "empty") so no CSVs are included in the data folder for them:   

- FairviewCrescent    
- Fraser River Parkade   
- Gerald McGavin UBC Rugby Centre   
- Health Sciences Parkade   
- Horticulture   
- Marine Drive Res 2 (all Marine Drive info might currently be in Marine Drive Res 1?)   
- Marine Drive Res 3   
- Marine Drive Res 4   
- Marine Drive Res 5   
- Marine Drive Res 6   
- North Parkade   
- Old Admin
- Point Grill   
- Rose Garden Parkade   
- Rugby Centre   
- Rugby Pavillion
- Thunderbird Park   
- Thunderbird Parkade   
- Vanier Complex (maybe has been split into indv. Vanier entries?)   
- West Parkade   


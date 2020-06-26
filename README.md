# Data 599 - Capstone


## File Directory
- [code](code/) - all code is found in this folder:
  - aggregation.py (various aggregation-related functions called by main.py)
  - cluster.py (various clustering functions called by main.py)
  - data_preparation.py (various functions for querying and cleaning data, called by main.py)
  - feature_selection.py (stand alone code used to perform feature selection - resulting feature selection choices were hard-coded into main.py)
  - **main.py (the primary file that drives the program)**
  - write_enduse_to_influx.py (stand alone code meant to be run after main.py to write the results into the InfluxDB)
  - classification_model_comparison_and_selection_tool.ipynb (Jupyter Notebook designed to compare classification models)
  - example_notebook.ipynb (Jupyter Notebook demonstrating the use of main.py and write_enduse_to_influx.py, full of annotations and explanations in markdown cells)
- [data](data/) -  contains files used demonstrating functionality in the example_notebook and also the files that are used in main.py (see the constants section of the code for a full listing)
- [Final_Report](Final_Report/) - Final report for capstone project
- [Logistics](Logistics/) - Contains a variety of documents used during the capstone project to communicate information within the team and track progress
- [Meetings](Meetings/) - Meeting agendas and minutes for capstone project
- [misc_work](misc_work/) - Contains an (slightly tangential) investigation into which fields/columns can be found in every building in the EWS SkySpark database
- [onepage_summary](onepage_summary/) - Contains document summarizing capstone project, was provided in advance of final presentation
- [proposal](proposal/) - Contains original capstone project proposal
- [visualization](visualization/) - Contains markdown document with details on the Grafana visualization dashboard that was created and .json file export of the dashboard.   
- [Weekly_Presentations](Weekly_Presentations/) - Contains PDF copies of each week's capstone project presentation **(Including Final Presentation)**
- [Work_Logs](Work_Logs/) - Contains individual and team work logs for the capstone project


## Project Description
The UBC Urban Data Lab (UDL) was established to provide open access to sustainability data. UDL provides access to an InfluxDB time series database that contains data on instruments that record the power, energy, water, and gas use of each UBC building. Currently, many instruments have descriptive tags that are not understandable or are too granular for practical use by building managers. For that reason, the focus of the project is to apply machine learning techniques to classify and group instruments by energy end-use. This information will be useful for building managers to easily identify where energy efficiency improvements can be made.
The proposed project was to create a Python program that queries, cleans, and classifies each instrument that meters energy consumption - referred to as an Energy Consumption (EC) sensor in this project - by end-use type. While these EC sensors only make up a small percentage of the total instruments, data from the remaining instruments - referred to as Non-Energy Consumption (NC) sensors - was used to assist with the end-use classification task. The program queried EC and NC data from InfluxDB and the data was fed into a series of models. First, NC data was clustered into groups that reacted similarly. Next, the clustered NC data was used to model the EC/NC relationship. The purpose of modeling the relationship was to develop feature engineering for the EC end-use classification. Afterwards, the model coefficients, instrument metadata, aggregated EC data, and hand-labeled end-use training data were joined together. The final data set was fed into a classification model where EC sensors with unknown end-uses were classified. 
As seen in the Table 1 below, the classification model was able to predict and classify all 208 unknown energy consumption instruments into end-use categories.

Table 1: Number of Sensors per End-Use
| End-Use Label | Sensor Count | % of Sensors  | 
| ------------- | ------------- | ------------- |
|00_HEATING_SPACE_AND_WATER|54|26%|
|01_SPACE_COOLING|35|17%|
|02_HEATING_COOLING_COMBINED|39|19%|
|03_LIGHTING_NORMAL|26|13%|
|04_LIGHTING_EMERGENCY|10|5%|
|05_OTHER|44|21%|
|**Total**|**208**|**100%**|


The project achieved its goal by delivering a Python program that queries, cleans, and classifies instruments by end-use from live-streaming InfluxDB data for the Pharmacy building. The prediction accuracy of the program when applied to the testing dataset was 94.3%. The Python program assists with UDL’s vision of assisted Artificial Intelligence (AI) for proactive and preventive maintenance. A few recommendations for future work include increasing the size of the labeled training set and modifying code to work with the updated UDL database. 

## Project Approach
![](Logistics/Diagrams/Project_Approach.png)

## Project Flowchart
![](Logistics/Diagrams/Project_Flowchart.png)

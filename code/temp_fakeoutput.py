# developing outside of main.py to make it easier to test. Will move into main.py once it is working.

import numpy as np
import pandas as pd
import aggregation

fake_step1_output = pd.read_csv("test_data/2020-03-16.csv") 
fake_step1_output["clust_group_num"] = np.random.randint(1, 5, fake_step1_output.shape[0])
fake_step1_output.drop(["equipRef","groupRef","navName","siteRef","typeRef","unit"], axis=1, inplace=True)
fake_step1_output = aggregation.agg_all(fake_step1_output)
print(fake_step1_output.head())
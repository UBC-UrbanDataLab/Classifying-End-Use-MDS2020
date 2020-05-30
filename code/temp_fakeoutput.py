# developing outside of main.py to make it easier to test. Will move into main.py once it is working.

import numpy as np
import pandas as pd
import aggregation

fake_step1_output = pd.concat([pd.read_csv("test_data/2020-03-16.csv"),pd.read_csv("test_data/2020-05-01.csv")]) 
fake_step1_output["clust_group_num"] = np.random.randint(1, 5, fake_step1_output.shape[0])
fake_step1_output = aggregation.split_datetime(fake_step1_output)
fake_step1_output.drop(["equipRef","groupRef","navName","siteRef","typeRef","unit", "month"], axis=1, inplace=True)

fake_step1_output = aggregation.agg_all(fake_step1_output, num_how="mean", col_idx=[2,4,3],last_idx_to_col=False)
fake_step1_output = fake_step1_output.assign(stddev_value=np.random.uniform(1, 5, fake_step1_output.shape[0]),
min_value=np.random.uniform(1, 5, fake_step1_output.shape[0]),
max_value=np.random.uniform(1, 5, fake_step1_output.shape[0]),
mean_update_rate=np.random.uniform(1, 5, fake_step1_output.shape[0]))
fake_step1_output.rename(columns={"value":"mean_value"}, inplace=True)
print(fake_step1_output.head(25))

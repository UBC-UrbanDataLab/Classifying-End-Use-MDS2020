import unittest
import pandas as pd
import up_only
class UpdateRateAggregationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.test_df = pd.read_csv("test_data/numSensorUpdatesApr01ToMay20.csv")
    def test_agg_up_rate(self):
        results = up_only.agg_up_rate(self.test_df)
        self.assertEqual(up_only.agg_up_rate("Garbage In"), 0) #should return 0 if input not a dataframe
        self.assertEqual(up_only.agg_up_rate(self.test_df[["groupRef","typeRef"]]),0) #should return 0 if dataframe missing expected columns

if __name__ == "__main__":
    unittest.main()
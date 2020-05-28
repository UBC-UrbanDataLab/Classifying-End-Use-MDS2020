import unittest

import pandas as pd
import numpy as np

import data_preparation

class DataPrepFunctionsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_df = pd.read_csv("test_data/2020-05-01.csv")
    def test_get_data_type(self):
        self.assertEqual(data_preparation.get_data_type("True"), "bool")
        self.assertEqual(data_preparation.get_data_type("False"), "bool")
        self.assertEqual(data_preparation.get_data_type(True), "bool")
        self.assertEqual(data_preparation.get_data_type(False), "bool")
        self.assertEqual(data_preparation.get_data_type(123), "num")
        self.assertEqual(data_preparation.get_data_type(123.456), "num")
        self.assertEqual(data_preparation.get_data_type("123"), "num")
        self.assertEqual(data_preparation.get_data_type("123.456"), "num")
        self.assertEqual(data_preparation.get_data_type("123.456a"), "str")
        self.assertEqual(data_preparation.get_data_type("foobar"), "str")
        self.assertEqual(data_preparation.get_data_type(None), "str")
    def test_encode_categorical(self):
        #This is a place holder. Need to think of some good tests
        with self.assertRaises(AttributeError):
            data_preparation.encode_categorical("foobar")
        with self.assertRaises(AttributeError):
            data_preparation.encode_categorical(22)
        self.assertIsInstance(data_preparation.encode_categorical(self.test_df,[1]), np.ndarray)
        
if __name__ == "__main__":
    unittest.main()
import unittest

import pandas as pd
import numpy as np

import data_preparation

class DataPrepTestCase(unittest.TestCase):
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
    
    def test_separate_cat_and_cont(self):
        #Just the basic tests right now. Need some more indepth tests, maybe checking something like shape of returns?
        self.assertIsInstance(data_preparation.separate_cat_and_cont(self.test_df), tuple)
        for entry in data_preparation.separate_cat_and_cont(self.test_df):
            self.assertIsInstance(entry,pd.DataFrame)

    def test_encode_categorical(self):
        #This is a place holder. Need to think of some good tests
        with self.assertRaises(AttributeError):
            data_preparation.encode_categorical("foobar")
        with self.assertRaises(AttributeError):
            data_preparation.encode_categorical(22)
        self.assertIsInstance(data_preparation.encode_categorical(self.test_df,[1]), np.ndarray)
    
    #Need to troubleshoot this one
    #def test_scale_continuous(self):
        #This is a place holder. Need to think of some good tests
        #self.assertIsInstance(data_preparation.scale_continuous(data_preparation.separate_cat_and_cont(
        #    self.test_df)[1], indexes=[7]), np.ndarray)

    def test_encode_and_scale_values(self):
        #This is just one basic test as a place holder. Pretty large function, probably needs a lot of tests
        self.assertIsInstance(data_preparation.encode_and_scale_values(self.test_df), pd.DataFrame)

    def test_encode_units(self):
        #This is a place holder test, need to add some more indepth tests.
        self.assertIsInstance(data_preparation.encode_units(self.test_df), pd.DataFrame)

#class DataConnectionTestCase(unittest.TestCase):
    #@classmethod
    #def setUpClass(cls):
    #    cls.test_df = pd.read_csv("test_data/2020-05-01.csv")
    #def test_connect_to_db(self):
    #    This is probably a stupid test to include. 
    #    self.assertIsInstance(data_preparation.connect_to_db(database = 'SKYSPARK'), influxdb.)

if __name__ == "__main__":
    unittest.main()
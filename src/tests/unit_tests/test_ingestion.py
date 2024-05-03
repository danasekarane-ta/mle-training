import os
import unittest

import pandas as pd
from housingPricePrediction.ingest_pkg import data_ingestion


class TestDataIngestion(unittest.TestCase):
    """ Test class to test the Data ingestion module"""
    def test_fetch_housing_data(self):
        """ Test the Housing data function"""
        data_ingestion.fetch_housing_data()
        self.assertTrue(os.path.exists("datasets/housing/housing.csv"))

    def test_load_housing_data(self):
        """Test the load housing data"""
        data = data_ingestion.load_housing_data()
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 20640)


if __name__ == '__main__':
    unittest.main()

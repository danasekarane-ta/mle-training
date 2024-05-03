import os
import unittest

import pandas as pd

from housingPricePrediction import data_ingestion


class TestDataIngestion(unittest.TestCase):
    def test_feature_extraction(self):
        self.housing = pd.DataFrame({'total_rooms': [50, 100, 200],
                                     'households': [10, 20, 50],
                                     'total_bedrooms': [50, 100, 200],
                                     'population': [200, 500, 1000]})
        self.housing = data_ingestion.feature_extraction(self.housing)
        self.assertIn("population_per_household", self.housing.columns)
        self.assertListEqual(list(self.housing['rooms_per_household']),
                             [5, 5, 4])

    def test_imputing_data(self):
        self.housing = pd.DataFrame({'total_rooms': [50, None, 200],
                                     'households': [None, 20, 50],
                                     'total_bedrooms': [50, None, 200],
                                     'population': [200, 500, None],
                                     'median_house_value': [100, 200, 300],
                                     'ocean_proximity': ['Near Bay', 'Inland',
                                                         'NEAR OCEAN']})
        data, _, X_prepared = data_ingestion.imputing_data(self.housing)
        self.assertEqual(X_prepared.shape, (3, 4))
        self.assertListEqual(list(X_prepared['total_rooms']),
                             [50.0, 125.0, 200.0])

    def test_fetch_housing_data(self):
        data_ingestion.fetch_housing_data()
        self.assertTrue(os.path.exists("datasets/housing/housing.csv"))

    def test_load_housing_data(self):
        data = data_ingestion.load_housing_data()
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()

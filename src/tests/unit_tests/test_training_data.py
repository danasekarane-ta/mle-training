import unittest

import numpy as np
import pandas as pd

from housingPricePrediction.ingest_pkg import data_training


class TestTrainingModel(unittest.TestCase):
    """ Test class for training model class
    """
    def test_stratifiedShuffleSplit(self):
        """ Test that the suffle works
        """
        np.random.seed(42)
        self.data = {
            "income": np.random.randint(20000, 100000, 1000),
            "age": np.random.randint(20, 65, 1000),
        }
        self.data = pd.DataFrame(self.data)

        self.data["income_cat"] = pd.cut(self.data["income"],
                                         bins=[0, 30000, 60000, 90000, np.inf],
                                         labels=[1, 2, 3, 4])

        train_set, test_set, strain, stest = \
            data_training.stratified_Shuffle_Split(self.data)
        self.assertGreaterEqual(len(set(strain)), 1)
        self.assertEqual(train_set.shape[0], 800)
        self.assertEqual(test_set.shape[0], 200)


if __name__ == '__main__':
    unittest.main()

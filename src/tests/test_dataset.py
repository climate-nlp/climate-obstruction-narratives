import unittest
import pandas as pd


class TestDataset(unittest.TestCase):
    """
    Test for dataset
    """

    train_file = './Data/train.csv'
    dev_file = './Data/dev.csv'
    test_file = './Data/test.csv'

    label_list = [
        "CA",
        "CB",
        "GA",
        "GC",
        "PA",
        "PB",
        "SA",
    ]

    def test_id(self):
        trains = pd.read_csv(self.train_file)['id'].to_list()
        devs = pd.read_csv(self.dev_file)['id'].to_list()
        tests = pd.read_csv(self.test_file)['id'].to_list()

        # Validate sample overlaps
        self.assertEqual(len(trains), len(set(trains)))
        self.assertEqual(len(devs), len(set(devs)))
        self.assertEqual(len(tests), len(set(tests)))

        # Validate test data leakages
        self.assertFalse(set(trains) & set(tests))
        self.assertFalse(set(trains) & set(devs))
        self.assertFalse(set(devs) & set(tests))

    def test_body(self):
        trains = pd.read_csv(self.train_file)['ad_creative_body'].to_list()
        devs = pd.read_csv(self.dev_file)['ad_creative_body'].to_list()
        tests = pd.read_csv(self.test_file)['ad_creative_body'].to_list()

        # Validate sample overlaps
        self.assertEqual(len(trains), len(set(trains)))
        self.assertEqual(len(devs), len(set(devs)))
        self.assertEqual(len(tests), len(set(tests)))

        # Validate test data leakages
        self.assertFalse(set(trains) & set(tests))
        self.assertFalse(set(trains) & set(devs))
        self.assertFalse(set(devs) & set(tests))

    def test_label(self):

        def _count(labels, label_name):
            return len([l for l in labels if l == label_name])

        trains = pd.read_csv(self.train_file)['Typology 1'].dropna().to_list() \
            + pd.read_csv(self.train_file)['Typology 2'].dropna().to_list() \
            + pd.read_csv(self.train_file)['Typology 3'].dropna().to_list() \
            + pd.read_csv(self.train_file)['Typology 4'].dropna().to_list()
        devs = pd.read_csv(self.dev_file)['Typology 1'].dropna().to_list() \
            + pd.read_csv(self.dev_file)['Typology 2'].dropna().to_list() \
            + pd.read_csv(self.dev_file)['Typology 3'].dropna().to_list() \
            + pd.read_csv(self.dev_file)['Typology 4'].dropna().to_list()
        tests = pd.read_csv(self.test_file)['Typology 1'].dropna().to_list() \
            + pd.read_csv(self.test_file)['Typology 2'].dropna().to_list() \
            + pd.read_csv(self.test_file)['Typology 3'].dropna().to_list() \
            + pd.read_csv(self.test_file)['Typology 4'].dropna().to_list()

        # Validate label names
        self.assertFalse(set(trains) - set(self.label_list))
        self.assertFalse(set(devs) - set(self.label_list))
        self.assertFalse(set(tests) - set(self.label_list))

        # Validate label num
        self.assertEqual(_count(trains, 'CA'), 221)
        self.assertEqual(_count(trains, 'CB'), 166)
        self.assertEqual(_count(trains, 'GA'), 102)
        self.assertEqual(_count(trains, 'GC'), 59)
        self.assertEqual(_count(trains, 'PA'), 225)
        self.assertEqual(_count(trains, 'PB'), 32)
        self.assertEqual(_count(trains, 'SA'), 69)

        self.assertEqual(_count(devs, 'CA'), 30)
        self.assertEqual(_count(devs, 'CB'), 32)
        self.assertEqual(_count(devs, 'GA'), 6)
        self.assertEqual(_count(devs, 'GC'), 0)
        self.assertEqual(_count(devs, 'PA'), 70)
        self.assertEqual(_count(devs, 'PB'), 12)
        self.assertEqual(_count(devs, 'SA'), 21)

        self.assertEqual(_count(tests, 'CA'), 49)
        self.assertEqual(_count(tests, 'CB'), 28)
        self.assertEqual(_count(tests, 'GA'), 33)
        self.assertEqual(_count(tests, 'GC'), 56)
        self.assertEqual(_count(tests, 'PA'), 89)
        self.assertEqual(_count(tests, 'PB'), 3)
        self.assertEqual(_count(tests, 'SA'), 10)

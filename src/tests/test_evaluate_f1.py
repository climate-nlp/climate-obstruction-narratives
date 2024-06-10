import unittest
from typing import List
import pandas as pd
import sklearn.metrics as sk_metrics
import src.evaluate_f1


class TestEvaluateF1(unittest.TestCase):
    """
    Test class of src/evaluator_f1.py
    """

    gold_file = './Data/test.csv'
    prediction_file = './src/tests/test_predict_results.csv'
    label_list = [
        "CA",
        "CB",
        "GA",
        "GC",
        "PA",
        "PB",
        "SA",
    ]

    def open_for_sklearn(self, file_path: str) -> List:
        df = pd.read_csv(file_path)
        data_points = []
        for i, row in df.iterrows():
            labels = [row[l] for l in ['Typology 1', 'Typology 2', 'Typology 3', 'Typology 4']
                      if not pd.isna(row[l])]
            logits = [1 if l in labels else 0 for l in self.label_list]
            data_points.append(logits)
        return data_points

    def test_compare_with_sklearn(self):
        """
        Here, we compare results of our scorer and scoring with scikit-learn.
        If the both results are the same, our scorer works properly.
        """
        # Compute scores by our scorer
        res = src.evaluate_f1.evaluate_prediction_file(gold_file=self.gold_file, prediction_file=self.prediction_file)
        res = {(r['label'], r['level'], r['metric']): r['value'] for r in res}

        # Compute label-wise scores by sklearn
        y_true, y_pred = self.open_for_sklearn(self.gold_file), self.open_for_sklearn(self.prediction_file)
        sk_prec, sk_rec, sk_f, sk_supp = sk_metrics.precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred
        )

        # Label-wise evaluation
        for i, label in enumerate(self.label_list):
            self.assertAlmostEqual(
                src.evaluate_f1.process_score(sk_prec[i]),
                res[(label, 'sub_category', 'precision')]
            )
            self.assertAlmostEqual(
                src.evaluate_f1.process_score(sk_rec[i]),
                res[(label, 'sub_category', 'recall')]
            )
            self.assertAlmostEqual(
                src.evaluate_f1.process_score(sk_f[i]),
                res[(label, 'sub_category', 'f')]
            )
            self.assertAlmostEqual(
                sk_supp[i],
                res[(label, 'sub_category', 'n_gold')]
            )

        # Compute micro average scores by sklearn
        sk_micro_prec, sk_micro_rec, sk_micro_f, _ = sk_metrics.precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, average='micro'
        )
        self.assertAlmostEqual(
            src.evaluate_f1.process_score(sk_micro_prec),
            res[('Total', 'sub_category', 'precision')]
        )
        self.assertAlmostEqual(
            src.evaluate_f1.process_score(sk_micro_rec),
            res[('Total', 'sub_category', 'recall')]
        )
        self.assertAlmostEqual(
            src.evaluate_f1.process_score(sk_micro_f),
            res[('Total', 'sub_category', 'f')]
        )

    def test_process_score(self):
        self.assertEqual(
            src.evaluate_f1.process_score(0),
            0
        )
        self.assertEqual(
            src.evaluate_f1.process_score(1),
            100
        )
        self.assertEqual(
            src.evaluate_f1.process_score(0.1234),
            12.3
        )
        self.assertEqual(
            src.evaluate_f1.process_score(0.1),
            10.0
        )
        self.assertEqual(
            src.evaluate_f1.process_score(0.1205),
            12.0
        )
        self.assertEqual(
            src.evaluate_f1.process_score(0.1204),
            12.0
        )
        self.assertEqual(
            src.evaluate_f1.process_score(0.1206),
            12.0
        )
        self.assertEqual(
            src.evaluate_f1.process_score(0.12099999999),
            12.0
        )


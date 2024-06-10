import math
import argparse
import pandas as pd
from typing import Dict, List


def process_score(score: float) -> float:
    """
    Processes a probability score (score) by converting it into a scaled score out of 100,
    rounded down to a specified number of decimal places.

    Preconditions:
        - 0 <= score <= 1: The input score must be a normalized value between 0 and 1, inclusive.

    Postconditions:
        - The returned score is a float between 0 and 100, inclusive, rounded down to 1 decimal place.

    Parameters:
        score (float): The normalized score to process, must be between 0 and 1 inclusive.

    Returns:
        float: The processed score, scaled out of 100 and rounded down to 1 decimal place.
    """
    assert 0 <= score <= 1
    decimal_places = 1
    new_score = math.floor(score * 10 ** (2 + decimal_places)) / (10 ** decimal_places)
    assert 0 <= new_score <= 100
    return new_score


def calc_precision_recall_f1(
        golds: List[Dict], preds: List[Dict], pos_label: str = None
) -> Dict[str, float or int]:
    """
    Calculates the F1 score, along with precision and recall, for a given set of predicted
    and gold standard (true) samples. The function supports evaluation
    on all labels or a specific positive label.

    Preconditions:
        - golds and preds must be lists of dictionaries with 'id' and 'labels' keys.
        - Each 'id' must uniquely identify the data sample.
        - pos_label, if specified, must be a string representing the label of interest.

    Postconditions:
        - The function returns a dictionary with keys 'n_gold' (the number of gold label instances),
          'n_system' (the number of predicted label instances), 'n_correct' (the number of correct
          predictions), 'precision', 'recall', and 'f' (F1 score), with the latter three possibly
          processed by an external function for rounding or formatting.

    Parameters:
        golds (List[Dict]): A list of instances representing the gold standard.
        preds (List[Dict]): A list of instances representing the prediction.
        pos_label (str, optional): A string specifying the positive label to focus on. Defaults to None,
                                   indicating that all labels should be considered.

    Returns:
        Dict[str, Union[float, int]]: A dictionary containing the counts of gold labels, system
                                      (predicted) labels, correct predictions, and the calculated
                                      precision, recall, and F1 score. The values are either integers
                                      (for counts) or floats (for precision, recall, and F1).
    """
    g_tpls, s_tpls = set(), set()

    for gold in golds:
        for label in gold['labels']:
            if pos_label is None:
                g_tpls.add((gold['id'], label))
            elif label == pos_label:
                g_tpls.add((gold['id'],))

    for pred in preds:
        for label in pred['labels']:
            if pos_label is None:
                s_tpls.add((pred['id'], label))
            elif label == pos_label:
                s_tpls.add((pred['id'],))

    g = len(g_tpls)
    s = len(s_tpls)
    c = len(g_tpls & s_tpls)
    p = c / s if s != 0 else 0.
    r = c / g if g != 0 else 0.
    f = ((2 * p * r) / (p + r)) if p + r != 0 else 0.

    assert 0 <= p <= 1
    assert 0 <= r <= 1
    assert 0 <= f <= 1
    assert 0 <= f <= ((p + r) / 2) + 1e-7

    return {
        'n_gold': g, 'n_system': s, 'n_correct': c,
        'precision': process_score(p), 'recall':  process_score(r), 'f': process_score(f)
    }


def evaluate_prediction_file(gold_file: str, prediction_file: str) -> List[Dict]:
    """
    Evaluates prediction scores by comparing predicted labels against gold standard labels
    for a given dataset. The evaluation is done at both sub-category and super-category levels
    for a set of typologies.

    This function reads in two CSV files: one containing the gold standard labels ('gold_file')
    and the other containing the predictions ('prediction_file'). Each file should have an 'id' column
    to uniquely identify each item and columns for 'Typology 1', 'Typology 2', 'Typology 3',
    and 'Typology 4', which contain the labels for each item. The function then calculates
    precision, recall, and F1 scores for each label as well as overall (micro) scores across all labels.

    Labels in the gold and prediction files are first evaluated at their specific typology level
    (sub-category). Then, using a predefined mapping to super-categories, the evaluation is also
    performed at the super-category level. This allows for an analysis of both detailed and
    general prediction accuracy.

    Parameters:
        gold_file (str): The path to the CSV file containing the gold standard labels.
        prediction_file (str): The path to the CSV file containing the predicted labels.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents the evaluation
                    results for a specific label and level (sub-category or super-category).
                    Each dictionary contains the 'label', 'level' (either 'sub_category' or
                    'super_category'), 'metric' (either 'precision', 'recall', or 'F1'), and
                    the 'value' of the metric.
    """
    g_df = pd.read_csv(gold_file)
    golds = []
    for i, row in g_df.iterrows():
        golds.append({
            'id': row['id'],
            'labels': [row[l] for l in ['Typology 1', 'Typology 2', 'Typology 3', 'Typology 4'] if not pd.isna(row[l])]
        })

    s_df = pd.read_csv(prediction_file)
    preds = []
    for i, row in s_df.iterrows():
        preds.append({
            'id': row['id'],
            'labels': [row[l] for l in ['Typology 1', 'Typology 2', 'Typology 3', 'Typology 4'] if not pd.isna(row[l])]
        })

    labels = set()
    for gold in golds:
        labels |= set(gold['labels'])
    labels = sorted(labels)

    result = []

    # Subcategory-level result
    for label in labels:
        for key, score in calc_precision_recall_f1(golds=golds, preds=preds, pos_label=label).items():
            result.append({
                'label': label,
                'level': 'sub_category',
                'metric': key,
                'value': score,
            })
    for key, score in calc_precision_recall_f1(golds=golds, preds=preds, pos_label=None).items():
        result.append({
            'label': f'Total',
            'level': 'sub_category',
            'metric': key,
            'value': score,
        })

    # Super category-level result
    category_map = {
        "CA": 'C',
        "CB": 'C',
        "GA": 'G',
        "GC": 'G',
        "PA": 'P',
        "PB": 'P',
        "SA": 'S',
    }

    golds = [{'id': d['id'], 'labels': [category_map[l] for l in d['labels']]} for d in golds]
    preds = [{'id': d['id'], 'labels': [category_map[l] for l in d['labels']]} for d in preds]

    super_labels = sorted(set(category_map.values()))
    for label in super_labels:
        for key, score in calc_precision_recall_f1(golds=golds, preds=preds, pos_label=label).items():
            result.append({
                'label': label,
                'level': 'super_category',
                'metric': key,
                'value': score,
            })
    for key, score in calc_precision_recall_f1(golds=golds, preds=preds, pos_label=None).items():
        result.append({
            'label': 'Total',
            'level': 'super_category',
            'metric': key,
            'value': score,
        })

    return result


def main(args):
    result = evaluate_prediction_file(gold_file=args.g, prediction_file=args.s)
    pd.DataFrame(result).to_csv(args.o, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        type=str,
        help="The gold csv file",
    )
    parser.add_argument(
        "-s",
        type=str,
        help="The system prediction csv file",
    )
    parser.add_argument(
        "-o",
        type=str,
        help="The output file path",
    )
    args = parser.parse_args()
    main(args=args)

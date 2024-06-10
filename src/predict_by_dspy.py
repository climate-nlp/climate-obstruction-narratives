import os.path
import json
import pandas as pd
import argparse
import time
from typing import List

import dsp
import dspy
from dspy.teleprompt import LabeledFewShot


LABEL_LIST = [
    "CA",
    "CB",
    "GA",
    "GC",
    "PA",
    "PB",
    "SA",
]


class BasicClassifier(dspy.Signature):
    """Please label the following advert according to the described typology. Many adverts will not be relevant so please label them as X. We are looking for narratives specifically from the oil and gas sector.

Community & Resilience
    CA: Emphasizes how the oil and gas sector contributes to local and national economies through tax revenues, charitable efforts, and support for local businesses.
    CB: Focuses on the creation and sustainability of jobs by the oil and gas industry.

Green Innovation and Climate Solutions
    GA: Highlights efforts to reduce greenhouse gas emissions through internal targets, policy support, voluntary initiatives, and emissions reduction technologies.
    GC: Promotes "clean" or "green" fossil fuels as part of climate solutions.

Pragmatism/Pragmatic Energy mix (Power systems and manufactured goods)
    PA: Portrays oil and gas as essential, reliable, affordable, and safe energy sources critical for maintaining power systems.
    PB: Emphasizes the importance of oil and gas as raw materials for various non-power-related uses and manufactured goods.

Patriotic Energy mix
    SA: Stresses how domestic oil and gas production benefits the nation, including energy independence, energy leadership, and the idea of supporting American energy.

X. No relevant typology detected.

This task is a multi-label classification and can have up to four labels amongst CA, CB, GA, GC, PA, PB, and SA.
If X is labeled, no other labels are allowed.
For example, a label containing GA and GC should be answered ["GA", "GC"].
"""
    text = dspy.InputField(prefix='Advert:', desc='text')
    answer = dspy.OutputField(prefix='Answer:', format=dsp.format_answers,
                              desc='The predicted labels with a JSON list')


class CoTClassifier(dspy.Signature):
    """Please label the following advert according to the described typology. Many adverts will not be relevant so please label them as X. We are looking for narratives specifically from the oil and gas sector.

Community & Resilience
    CA: Emphasizes how the oil and gas sector contributes to local and national economies through tax revenues, charitable efforts, and support for local businesses.
    CB: Focuses on the creation and sustainability of jobs by the oil and gas industry.

Green Innovation and Climate Solutions
    GA: Highlights efforts to reduce greenhouse gas emissions through internal targets, policy support, voluntary initiatives, and emissions reduction technologies.
    GC: Promotes "clean" or "green" fossil fuels as part of climate solutions.

Pragmatism/Pragmatic Energy mix (Power systems and manufactured goods)
    PA: Portrays oil and gas as essential, reliable, affordable, and safe energy sources critical for maintaining power systems.
    PB: Emphasizes the importance of oil and gas as raw materials for various non-power-related uses and manufactured goods.

Patriotic Energy mix
    SA: Stresses how domestic oil and gas production benefits the nation, including energy independence, energy leadership, and the idea of supporting American energy.

X. No relevant typology detected.

This task is a multi-label classification and can have up to four labels amongst CA, CB, GA, GC, PA, PB, and SA.
If X is labeled, no other labels are allowed.
For example, a label containing GA and GC should be answered ["GA", "GC"].

Reasoning process for analysis:
    First, read the advert text to understand its main message.
    Next, identify the key themes presented in the advert. This includes looking for mentions of economic impact, job creation, environmental efforts, or patriotic messaging.
    Then, match these themes to the typologies listed above. Determine which of the typologies the themes of the advert align with.
    If the advert contains elements from multiple categories, determine the primary focus of the advert and choose the most fitting category.
    Finally, label the advert according to the most appropriate typology.
"""
    text = dspy.InputField(prefix='Advert:', desc='text')
    answer = dspy.OutputField(prefix='Answer:', format=dsp.format_answers,
                              desc='The predicted labels with a JSON list')


class FewshotModule(dspy.Module):
    def __init__(self, classifier_cls, dspy_cls):
        super().__init__()
        self.generate_answer = dspy_cls(classifier_cls)

    def forward(self, text):
        prediction = self.generate_answer(text=text)
        if hasattr(prediction, 'rationale'):
            return dspy.Prediction(text=text, rationale=prediction.rationale, answer=prediction.answer)
        else:
            return dspy.Prediction(text=text, answer=prediction.answer)


def read_fewshot_file(file_path: str) -> List[dspy.Example]:
    fewshot_df = pd.read_csv(file_path)
    trainset = []
    for _, row in fewshot_df.iterrows():
        labels = []
        for i in [1, 2, 3, 4]:
            if not pd.isna(row[f'Typology {i}']):
                labels.append(row[f'Typology {i}'])
        labels = sorted(labels)
        if not labels:
            labels = ['X']
        trainset.append(dspy.Example(text=row['ad_creative_body'], answer=json.dumps(labels)))
    return trainset


def prepare_generator(method: str, trainset: List[dspy.Example]):
    if method == 'basic':
        classifier = BasicClassifier
        dspy_cls = dspy.Predict
    elif method == 'cot':
        classifier = CoTClassifier
        dspy_cls = dspy.ChainOfThought
    else:
        assert False

    if trainset:
        generate_answer = LabeledFewShot().compile(
            FewshotModule(classifier_cls=classifier, dspy_cls=dspy_cls), trainset=trainset
        )
    else:
        generate_answer = dspy_cls(classifier)

    return generate_answer


def main(args):
    global LABEL_LIST

    using_local_model = False
    if 'mistralai' in args.model_name:
        using_local_model = True
        lm = dspy.HFClientVLLM(model=args.model_name, port=args.vllm_port, url="http://localhost")
    else:
        lm = dspy.OpenAI(model=args.model_name, api_key=args.api_key, max_tokens=args.max_tokens)

    dspy.settings.configure(lm=lm)

    if args.fewshot_file:
        trainset = read_fewshot_file(args.fewshot_file)
    else:
        trainset = []

    generate_answer = prepare_generator(method=args.method, trainset=trainset)

    test_df = pd.read_csv(args.test_file)

    cached_data_points = []
    cached_ids = []
    if os.path.exists(args.output):
        output_df = pd.read_csv(args.output, index_col=[0])
        cached_ids = output_df['id'].to_list()
        cached_data_points = output_df.to_dict(orient='records')

    for _, test_data_point in test_df.iterrows():
        _id = test_data_point['id']
        if _id in cached_ids:
            print(f'Skip inference for {_id} because it exists in the output file.')
            continue

        pred = generate_answer(text=test_data_point['ad_creative_body'])
        print(lm.inspect_history())

        try:
            if 'mistralai' in args.model_name:
                # Since mistral models produce multi answers, we only try to extract the first answer
                answer_line = pred.answer.split('\n')[0]
                pred_labels = [l for l in LABEL_LIST if l in answer_line]
            else:
                # For OpenAI GPTs, we just apply json.loads function to extract labels
                pred_labels = json.loads(pred.answer)
        except Exception as e:
            pred_labels = []

        print(f'Parsed (predicted) labels: {pred_labels}')

        pred_labels = [l for l in pred_labels if l in LABEL_LIST]
        pred_labels = sorted(pred_labels)[:4]
        pred_labels = pred_labels + [None] * (4 - len(pred_labels))

        cached_data_points.append({
            'id': _id,
            'ad_creative_body': test_data_point['ad_creative_body'],
            'Typology 1': pred_labels[0],
            'Typology 2': pred_labels[1],
            'Typology 3': pred_labels[2],
            'Typology 4': pred_labels[3],
            'used_model': args.model_name,
            'used_method': args.method,
            'fewshot_size': len(trainset),
            'generated_answer': pred.answer,
            'generated_rationale': pred.rationale if args.method == 'cot' else None,
        })
        cached_ids.append(_id)
        pd.DataFrame(cached_data_points).to_csv(args.output)

        if not using_local_model and args.sleep_at_inference > 0:
            time.sleep(args.sleep_at_inference)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--api_key',
        type=str,
        default='',
        required=True
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='gpt-3.5-turbo'
    )
    parser.add_argument(
        '--vllm_port',
        type=int,
        default=8000
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default=None,
        required=True
    )
    parser.add_argument(
        '--fewshot_file',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['basic', 'cot'],
        default='basic',
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        '--sleep_at_inference',
        type=int,
        default=2,
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=300
    )
    args = parser.parse_args()
    main(args=args)

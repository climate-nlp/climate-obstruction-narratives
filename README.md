# Predicting Narratives of Climate Obstruction in Social Media Advertising

This is a codebase to reproduce our paper "Predicting Narratives of Climate Obstruction in Social Media Advertising" accepted to Findings of ACL 2024.
This project is not intended for use by investors or practitioners.


## Setup

#### Recommended Python version

```
Python 3.9.5
```

#### Install packages:

```bash
pip install torch==2.0.0
pip install -r requirements.txt
```


## Dataset

This project contains a Facebook ad dataset built based on the work of [Holder et al. 2023](https://link.springer.com/article/10.1007/s10584-023-03494-4) (see the bottom for reference.) 
The NLP tailored datasets are stored as CSV files as follows:

```text
├── all.csv
├── rogl_data
│   ├── evaluate_results.csv
│   └── predict_results.csv
├── dev.csv
├── test.csv
├── test_hidden.csv
├── train.csv
├── train_fewshot-128.csv
├── train_fewshot-32.csv
├── train_fewshot-512.csv
└── train_fewshot-8.csv
```

```all.csv``` contains all the samples. 
```train.csv```, ```dev.csv```, and ```test.csv``` are for training, development, and test samples, respectively.
```test_hidden.csv``` does not contain label information. This can be used to predict labels avoiding the risk of "label leaking".
The files ```train_fewshot-*.csv``` are for low-resource experiments (i.e., files for few-shot learning).
The ```rogl_data``` directory contains "silver" samples labeled by GPT-4 Turbo. They are used in the RoGL experiment.


#### The label definitions

Our task uses seven labels provided by Holder et al. 2023 as follows:

- CA: Emphasizes how the oil and gas sector contributes to local and national economies through tax revenues, charitable efforts, and support for local businesses.
- CB: Focuses on the creation and sustainability of jobs by the oil and gas industry.
- GA: Highlights efforts to reduce greenhouse gas emissions through internal targets, policy support, voluntary initiatives, and emissions reduction technologies.
- GC: Promotes "clean" or "green" fossil fuels as part of climate solutions.
- PA: Portrays oil and gas as essential, reliable, affordable, and safe energy sources critical for maintaining power systems.
- PB: Emphasizes the importance of oil and gas as raw materials for various non-power-related uses and manufactured goods.
- SA: Stresses how domestic oil and gas production benefits the nation, including energy independence, energy leadership, and the idea of supporting American energy.

See Holder et al. 2023's paper and our paper for more detailed description and statistics.


## Important codes

#### src/bert_model.py
This is to fine-tune and predict labels by BERT family models. 
The example of fine-tuning is:

```bash
  python src/bert_model.py \
    --model_name_or_path "bert-base-cased" \
    --output_dir ./output_dir \
    --train_file ./Data/train.csv \
    --do_train \
    --seed 42 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --max_steps 1000 \
    --max_seq_length 256 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_total_limit 1 \
    --save_steps 1000 --logging_steps 5 \
    --fp16
```

The example of prediction is:

```bash
python src/bert_model.py \
    --model_name_or_path <PATH TO THE TRAINED MODEL> \
    --output_dir ./output_dir \
    --test_file ./Data/test_hidden.csv \
    --do_predict \
    --seed 42 \
    --max_seq_length 256 \
    --per_device_eval_batch_size 16
```


#### src/predict_by_dspy.py
This is to predict labels by LLM zeroshot (or few-shot) prompting using [Stanford DSPy](https://github.com/stanfordnlp/dspy). The example usage is:

```bash
 python src/predict_by_dspy.py \
    --api_key <YOUR OPENAI API KEY> \
    --model_name "gpt-4-1106-preview" \
    --test_file ./Data/test_hidden.csv \
    --output ./output_dir/predict_results.csv \
    --method "basic" \
    --max_tokens 30
```

#### src/evaluate_f1.py
This is a script to evaluate the F-scores of the model output. The example usage is:

```bash
  python src/evaluate_f1.py \
    -s Result/bert-base-cased_full_seed-0/predict_results.csv \
    -g Data/test.csv \
    -o evaluation_result.csv
 ```


## Use trained model

The fine-tuned RoBERTa-large model can be obtained through [Hugging Face Hub](https://huggingface.co/climate-nlp/climate-obstructive-narratives).
You can use the model ```climate-nlp/climate-obstructive-narratives``` as follows:

```bash
python src/bert_model.py \
    --model_name_or_path "climate-nlp/climate-obstructive-narratives" \
    --output_dir ./output_dir \
    --test_file ./Data/test_hidden.csv \
    --do_predict \
    --seed 42 \
    --max_seq_length 256 \
    --per_device_eval_batch_size 16
```



## Run reproducing experiments

The results obtained by our experiments can be found in the ```Result.zip``` under the project root directory. 
You can try training and prediction on your own device as follows.
Please note that results may be slightly different from ours because of device or software environment differences.

### BERT and RoBERTa models

```bash
# BERT-base (low-resource experiments are also conducted.)
./experiments/run_bert_base.pbs
# RoBERTa-base (low-resource experiments are also conducted.)
./experiments/run_roberta_base.pbs
# RoBERTa-large (low-resource experiments are also conducted.)
./experiments/run_roberta_large.pbs
```

### RoGL
```bash
# RoBERTa-large + silver label training, Low-resource experiments
./experiments/run_rogl.pbs
```


### OpenAI GPT-4 Turbo and GPT-3.5 Turbo

At first, you have to edit APIKEY in the ```./scripts/run_gpt3-5-turbo.sh``` and ```./scripts/run_gpt4-turbo.sh``` so that you can use your own API key of OpenAI.

Then, run the following:

```bash
# GPT-3.5-turbo
./scripts/run_gpt3-5-turbo.sh
# GPT-4-turbo
./scripts/run_gpt4-turbo.sh
```


### Mistral

At first, you need to setup vLLM (https://github.com/vllm-project/vllm) under another project directory

#### Run on Mistral-7B-Instruct-v0.1
1. At the vLLM project, start the server: ```python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.1 --port 8000```
2. At our project, run ```./experiments/run_mistral-7b-instruct.sh```


## Run tests

We validate our scoring script and dataset by running the following script:
```bash
./scripts/run_tests.sh
```


## License

This is a collaborative project of [InfluenceMap](https://influencemap.org/) and Stanford. 
The rights to the dataset belong to InfluenceMap and the source code is joint property.
See [Terms and Conditions](https://influencemap.org/terms) of InfluenceMap.
See LICENSE file under this project for the license of the code (i.e., Apache 2.0).
Note that this project includes third party codes which are not under our license.
Different licenses apply to models and datasets.

- ```src/bert_model.py``` was obtained from [Hugging Face Transformers](https://github.com/huggingface/transformers) and modified to our use.


## Citation

Our project uses dataset based on a valuable annotation by [Holder et al. 2023](https://link.springer.com/article/10.1007/s10584-023-03494-4). When you use data of this project, do not forget to cite their paper:

```
@Article{holder-etal-2023,
    author={Holder, Faye
    and Mirza, Sanober
    and {Namson-Ngo-Lee}
    and Carbone, Jake
    and McKie, Ruth E.},
    title={Climate obstruction and Facebook advertising: how a sample of climate obstruction organizations use social media to disseminate discourses of delay},
    journal={Climatic Change},
    year={2023},
    month={Feb},
    day={10},
    volume={176},
    number={2},
    pages={16},
    issn={1573-1480},
    doi={10.1007/s10584-023-03494-4},
    url={https://doi.org/10.1007/s10584-023-03494-4}
}
```

The citation of our project is:

```
@inproceedings{rowlands-etal-2024-predicting,
    title = "Predicting Narratives of Climate Obstruction in Social Media Advertising",
    author = "Rowlands, Harri  and
      Morio, Gaku  and
      Tanner, Dylan  and
      Manning, Christopher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
}
```

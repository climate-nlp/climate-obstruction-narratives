set -eu

export APIKEY="dummy"
export MODEL="mistralai/Mistral-7B-Instruct-v0.1"

export TEST_FILE=./Data/test.csv
export PREDICT_FILE=./Data/test_hidden.csv

LOG_PREFIX=./Result/mistral-7b-instruct-v0.1

for METHOD in "basic"
do
  export METHOD=${METHOD}
  export FEWSHOT_FILE=""
  export LOGDIR=${LOG_PREFIX}_${METHOD}_zeroshot
  export MAX_TOKENS=300
  ./scripts/run_gpt.sh

  for FEWSHOT_NUM in 8 32
  do
      export LOGDIR=${LOG_PREFIX}_${METHOD}_fewshot-${FEWSHOT_NUM}
      export FEWSHOT_FILE=./Data/train_fewshot-${FEWSHOT_NUM}.csv
      export MAX_TOKENS=300
      ./scripts/run_gpt.sh
  done
done

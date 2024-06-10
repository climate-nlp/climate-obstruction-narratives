set -eu

export APIKEY="<your API key of OpenAI>"
export MODEL="gpt-4-1106-preview"

export TEST_FILE=./Data/test.csv
export PREDICT_FILE=./Data/test_hidden.csv

LOG_PREFIX=./Result/${MODEL}

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

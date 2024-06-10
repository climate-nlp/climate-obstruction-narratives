set -eu

# Need to export following envs
# APIKEY
# MODEL
# LOGDIR
# TEST_FILE
# PREDICT_FILE
# METHOD
# MAX_TOKENS
# [Optional] FEWSHOT_FILE

mkdir -p ${LOGDIR}

if [[ -z "${FEWSHOT_FILE}" ]]; then
  python src/predict_by_dspy.py \
    --api_key ${APIKEY} \
    --model_name ${MODEL} \
    --test_file ${PREDICT_FILE} \
    --output ${LOGDIR}/predict_results.csv \
    --method ${METHOD} \
    --max_tokens ${MAX_TOKENS}
else
  python src/predict_by_dspy.py \
    --api_key ${APIKEY} \
    --model_name ${MODEL} \
    --test_file ${PREDICT_FILE} \
    --output ${LOGDIR}/predict_results.csv \
    --method ${METHOD} \
    --fewshot_file ${FEWSHOT_FILE} \
    --max_tokens ${MAX_TOKENS}
fi

if [ ! -e ${LOGDIR}/evaluate_results.csv ]; then
  python src/evaluate_f1.py \
    -s ${LOGDIR}/predict_results.csv \
    -g ${TEST_FILE} \
    -o ${LOGDIR}/evaluate_results.csv
fi

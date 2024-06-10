set -eu

# Hyper-parameters ---------

if [[ -z "${TRAIN_FILE}" ]]; then
  echo "Environment variable TRAIN_FILE is not defined"
  exit
fi
if [[ -z "${TEST_FILE}" ]]; then
  echo "Environment variable TEST_FILE is not defined"
  exit
fi
if [[ -z "${PREDICT_FILE}" ]]; then
  echo "Environment variable PREDICT_FILE is not defined"
  exit
fi
if [[ -z "${LOGDIR}" ]]; then
  echo "Environment variable LOGDIR is not defined"
  exit
fi
if [[ -z "${SEED}" ]]; then
  echo "Environment variable SEED is not defined"
  exit
fi
if [[ -z "${PLM}" ]]; then
  echo "Environment variable PLM is not defined"
  exit
fi
if [[ -z "${BS}" ]]; then
  echo "Environment variable BS is not defined"
  exit
fi
if [[ -z "${GS}" ]]; then
  echo "Environment variable GS is not defined"
  exit
fi
if [[ -z "${MAX_TRAIN_LEN}" ]]; then
  echo "Environment variable MAX_TRAIN_LEN is not defined"
  exit
fi
if [[ -z "${MAX_INFER_LEN}" ]]; then
  echo "Environment variable MAX_INFER_LEN is not defined"
  exit
fi

LR=1e-5
WARMUP_RATIO=0.1
# Inference batch-size
EBS=8
# --------------------------


mkdir -p ${LOGDIR}


# -----------------------------------------
# 1. Fine-tuning
# -----------------------------------------


if [ ! -e ${LOGDIR}/pytorch_model.bin ]; then
  echo "Fine-tuning"
  python src/bert_model.py \
    --model_name_or_path ${PLM} \
    --output_dir ${LOGDIR} \
    --train_file ${TRAIN_FILE} \
    --do_train \
    --seed ${SEED} \
    --learning_rate ${LR} \
    --warmup_ratio ${WARMUP_RATIO} \
    --max_steps 1000 \
    --max_seq_length ${MAX_TRAIN_LEN} \
    --per_device_train_batch_size ${BS} \
    --gradient_accumulation_steps ${GS} \
    --save_strategy "steps" \
    --save_total_limit 1 \
    --save_steps 1000 --logging_steps 5 \
    --fp16
fi


# -----------------------------------------
# 2. Prediction and evaluation
# -----------------------------------------

if [ ! -e ${LOGDIR}/predict_results.csv ]; then
  echo "Predicting"
  python src/bert_model.py \
    --model_name_or_path ${LOGDIR} \
    --output_dir ${LOGDIR} \
    --test_file ${PREDICT_FILE} \
    --do_predict \
    --seed ${SEED} \
    --max_seq_length ${MAX_INFER_LEN} \
    --per_device_eval_batch_size ${EBS}
fi

if [ ! -e ${LOGDIR}/evaluate_results.csv ]; then
  echo "Evaluating"
  python src/evaluate_f1.py \
    -s ${LOGDIR}/predict_results.csv \
    -g ${TEST_FILE} \
    -o ${LOGDIR}/evaluate_results.csv
fi

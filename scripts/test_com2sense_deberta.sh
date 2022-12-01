DATA_DIR="datasets/com2sense"
#MODEL_TYPE="bert-base-cased"
#MODEL_TYPE="roberta-base"
MODEL_TYPE="microsoft/deberta-base"
TRAINED_MODEL="outputs/com2sense/ckpts/checkpoint-5000"
TASK_NAME="com2sense"
OUTPUT_DIR=${TASK_NAME}/ckpts


CUDA_VISIBLE_DEVICES=0 python3 -m trainers.train \
  --model_name_or_path ${TRAINED_MODEL} \
  --config_name "${TRAINED_MODEL}/config.json" \
  --output_dir "${OUTPUT_DIR}/ckpts" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --eval_split "test" \
  --tokenizer_name ${MODEL_TYPE} \
  --max_seq_length 128 \
  --do_eval \


  #--overwrite_output_dir \

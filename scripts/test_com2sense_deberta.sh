DATA_DIR="datasets/com2sense"
#MODEL_TYPE="bert-base-cased"
#MODEL_TYPE="roberta-base"
MODEL_TYPE="microsoft/deberta-base"
TRAINED_MODEL="outputs/com2sense/ckpts/checkpoint-5000"
TASK_NAME="com2sense"
OUTPUT_DIR=${TASK_NAME}/ckpts


CUDA_VISIBLE_DEVICES=0 python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --config_name "${MODEL_TYPE}" \
  --output_dir "${OUTPUT_DIR}/ckpts" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --eval_split "test" \
  --tokenizer_name ${MODEL_TYPE} \
  --max_seq_length 128 \
  --do_eval \
  --iters_to_eval 5000 \
  #--overwrite_output_dir \

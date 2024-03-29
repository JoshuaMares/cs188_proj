TASK_NAME="com2sense"
DATA_DIR="datasets/com2sense"
MODEL_TYPE="microsoft/deberta-base"
#MODEL_TYPE="bert-base-cased"
#MODEL_TYPE="roberta-base"
MODEL_TYPE="microsoft/deberta-base"

OUTPUT_DIR=${TASK_NAME}/ckpts


CUDA_VISIBLE_DEVICES=0 python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --config_name "${MODEL_TYPE}" \
  --output_dir "${OUTPUT_DIR}" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --eval_split "test" \
  --tokenizer_name ${MODEL_TYPE} \
  --max_seq_length 128 \
  --do_eval \
  --iters_to_eval 11000 \
  #--overwrite_output_dir \

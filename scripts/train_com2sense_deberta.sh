TASK_NAME="com2sense"
DATA_DIR="datasets/com2sense"
MODEL_TYPE="microsoft/deberta-base"
OUTPUT_DIR=${TASK_NAME}/ckpts


CUDA_VISIBLE_DEVICES=0 python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --eval_all_checkpoints \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 1 \
  --learning_rate 1e-5 \
  --max_steps 5000 \
  --max_seq_length 128 \
  --output_dir "${OUTPUT_DIR}" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --save_steps 1000 \
  --logging_steps 1000 \
  --warmup_steps 100 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --do_not_load_optimizer \
  --overwrite_output_dir \

DATA_DIR="datasets/semeval_2020_task4"
#MODEL_TYPE="bert-base-cased"
#MODEL_TYPE="roberta-base"
MODEL_TYPE="microsoft/deberta-base"
TASK_NAME="semeval"
OUTPUT_DIR=${TASK_NAME}

CUDA_VISIBLE_DEVICES=0 python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --eval_all_checkpoints \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 1 \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --max_seq_length 128 \
  --output_dir "${OUTPUT_DIR}/ckpts" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --save_steps 500 \
  --logging_steps 100 \
  --warmup_steps 100 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --do_not_load_optimizer \
  --evaluate_during_training \
  --overwrite_output_dir \

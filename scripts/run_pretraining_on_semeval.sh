TASK_NAME="com2sense"
DATA_DIR="datasets/semeval_2020_task4"
MODEL_TYPE="microsoft/deberta-base"

python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_not_load_optimizer \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 1 \
  --learning_rate 1e-5 \
  --max_steps 5000 \
  --max_seq_length 128 \
  --output_dir "pretrain/ckpts" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 1000 \
  --logging_steps 1000 \
  --max_eval_steps 1000 \
  --warmup_steps 100 \
  --eval_split "dev" \
  --score_average_method "micro" \
  --overwrite_output_dir \
  --training_phase "pretrain" \
  --seed 42 \

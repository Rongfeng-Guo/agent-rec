export CUDA_VISIBLE_DEVICES=2
# Define hyperparameters
LEARNING_RATE=5.0e-6
NUM_EPOCHS=3.0
SFT_BETA=0
DATA_ROLE="recommend,search" # 可选: search, ask, recommend, all (多选用逗号分隔)
ADPO_KL_WEIGHT=1.0
# Define the output directory dynamically based on learning rate and num_epochs
OUTPUT_DIR="/home/u2023000157/ecpo/fine-tuning/sto_adpo_v1_sample1_${DATA_ROLE}/Game/lr${LEARNING_RATE}_epochs${NUM_EPOCHS}_sft${SFT_BETA}_adpo_kl${ADPO_KL_WEIGHT}"

# Create the output directory if it doesn't existz
mkdir -p $OUTPUT_DIR

# Run the training command directly with command-line arguments, without using YAML
llamafactory-cli train \
  --model_name_or_path /home/share/pretrain_models/Meta-Llama-3.1-8B-Instruct \
  --adapter_name_or_path /home/u2023000157/ecpo/fine-tuning/sft/Game/lr5.0e-5_epochs3.0\
  --ref_model /home/share/pretrain_models/Meta-Llama-3.1-8B-Instruct \
  --ref_model_adapters /home/u2023000157/ecpo/fine-tuning/sft/Game/lr5.0e-5_epochs3.0 \
  --stage dpo \
  --do_train true \
  --pref_loss adpo \
  --adpo_kl_type abs \
  --pref_ftx $SFT_BETA \
  --adpo_kl_weight $ADPO_KL_WEIGHT \
  --finetuning_type lora \
  --lora_target all \
  --dataset game_sto_dpo_v1_sample1 \
  --template llama3 \
  --train_data_role $DATA_ROLE \
  --cutoff_len 2048 \
  --overwrite_cache true \
  --preprocessing_num_workers 16 \
  --output_dir $OUTPUT_DIR \
  --logging_steps 10 \
  --save_steps 500 \
  --plot_loss true \
  --overwrite_output_dir true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $NUM_EPOCHS \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 true \
  --ddp_timeout 180000000 \
  --val_size 0.1 \
  --per_device_eval_batch_size 1 \
  --eval_strategy steps \
  --eval_steps 500 \
  > $OUTPUT_DIR/train.log 2>&1


# Define hyperparameters
NUM_EPOCHS=3.0
SFT_BETA=0
DATA_ROLE="ask,recommend" # 可选: search, ask, recommend, all (多选用逗号分隔)
ADPO_KL_WEIGHT=1.0
# Define the output directory dynamically based on learning rate and num_epochs
OUTPUT_DIR="/home/u2023000157/ecpo/fine-tuning/sto_adpo_v1_sample1_${DATA_ROLE}/Game/lr${LEARNING_RATE}_epochs${NUM_EPOCHS}_sft${SFT_BETA}_adpo_kl${ADPO_KL_WEIGHT}"

# Create the output directory if it doesn't existz
mkdir -p $OUTPUT_DIR

# Run the training command directly with command-line arguments, without using YAML
llamafactory-cli train \
  --model_name_or_path /home/share/pretrain_models/Meta-Llama-3.1-8B-Instruct \
  --adapter_name_or_path /home/u2023000157/ecpo/fine-tuning/sft/Game/lr5.0e-5_epochs3.0\
  --ref_model /home/share/pretrain_models/Meta-Llama-3.1-8B-Instruct \
  --ref_model_adapters /home/u2023000157/ecpo/fine-tuning/sft/Game/lr5.0e-5_epochs3.0 \
  --stage dpo \
  --do_train true \
  --pref_loss adpo \
  --adpo_kl_type abs \
  --pref_ftx $SFT_BETA \
  --adpo_kl_weight $ADPO_KL_WEIGHT \
  --finetuning_type lora \
  --lora_target all \
  --dataset game_sto_dpo_v1_sample1 \
  --template llama3 \
  --train_data_role $DATA_ROLE \
  --cutoff_len 2048 \
  --overwrite_cache true \
  --preprocessing_num_workers 16 \
  --output_dir $OUTPUT_DIR \
  --logging_steps 10 \
  --save_steps 500 \
  --plot_loss true \
  --overwrite_output_dir true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $NUM_EPOCHS \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 true \
  --ddp_timeout 180000000 \
  --val_size 0.1 \
  --per_device_eval_batch_size 1 \
  --eval_strategy steps \
  --eval_steps 500 \
  > $OUTPUT_DIR/train.log 2>&1

# Define hyperparameters
NUM_EPOCHS=3.0
SFT_BETA=0
DATA_ROLE="ask,search" # 可选: search, ask, recommend, all (多选用逗号分隔)
ADPO_KL_WEIGHT=1.0
# Define the output directory dynamically based on learning rate and num_epochs
OUTPUT_DIR="/home/u2023000157/ecpo/fine-tuning/sto_adpo_v1_sample1_${DATA_ROLE}/Game/lr${LEARNING_RATE}_epochs${NUM_EPOCHS}_sft${SFT_BETA}_adpo_kl${ADPO_KL_WEIGHT}"

# Create the output directory if it doesn't existz
mkdir -p $OUTPUT_DIR

# Run the training command directly with command-line arguments, without using YAML
llamafactory-cli train \
  --model_name_or_path /home/share/pretrain_models/Meta-Llama-3.1-8B-Instruct \
  --adapter_name_or_path /home/u2023000157/ecpo/fine-tuning/sft/Game/lr5.0e-5_epochs3.0\
  --ref_model /home/share/pretrain_models/Meta-Llama-3.1-8B-Instruct \
  --ref_model_adapters /home/u2023000157/ecpo/fine-tuning/sft/Game/lr5.0e-5_epochs3.0 \
  --stage dpo \
  --do_train true \
  --pref_loss adpo \
  --adpo_kl_type abs \
  --pref_ftx $SFT_BETA \
  --adpo_kl_weight $ADPO_KL_WEIGHT \
  --finetuning_type lora \
  --lora_target all \
  --dataset game_sto_dpo_v1_sample1 \
  --template llama3 \
  --train_data_role $DATA_ROLE \
  --cutoff_len 2048 \
  --overwrite_cache true \
  --preprocessing_num_workers 16 \
  --output_dir $OUTPUT_DIR \
  --logging_steps 10 \
  --save_steps 500 \
  --plot_loss true \
  --overwrite_output_dir true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $NUM_EPOCHS \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 true \
  --ddp_timeout 180000000 \
  --val_size 0.1 \
  --per_device_eval_batch_size 1 \
  --eval_strategy steps \
  --eval_steps 500 \
  > $OUTPUT_DIR/train.log 2>&1

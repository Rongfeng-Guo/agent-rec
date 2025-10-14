#!/bin/bash

# Set CUDA devices to be used for training
export CUDA_VISIBLE_DEVICES=1,2

# Define hyperparameters
LEARNING_RATE=1.0e-5  # Set the learning rate
NUM_EPOCHS=3.0         # Set the number of epochs

# Define the output directory dynamically based on learning rate and number of epochs
OUTPUT_DIR="/data/fxy/ecpo/fine-tuning/ppo/Book/lr${LEARNING_RATE}_epochs${NUM_EPOCHS}"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the training command directly with command-line arguments
nohup llamafactory-cli train \
  --model_name_or_path /data/pretrain_dir/Llama-3.1 \
  --adapter_name_or_path /data/fxy/ecpo/fine-tuning/sft/Book/lr5.0e-5_epochs3.0 \
  --reward_model /data/fxy/ecpo/fine-tuning/reward \
  --stage ppo \
  --do_train true \
  --finetuning_type lora \
  --lora_target all \
  --dataset amazon_book_ppo \
  --template llama3 \
  --cutoff_len 2048 \
  --overwrite_cache true \
  --preprocessing_num_workers 16 \
  --output_dir $OUTPUT_DIR \
  --logging_steps 10 \
  --save_steps 500 \
  --plot_loss true \
  --overwrite_output_dir true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $NUM_EPOCHS \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 true \
  --ddp_timeout 180000000 \
  --max_new_tokens 512 \
  --top_k 0 \
  --top_p 0.9 \
  > $OUTPUT_DIR/train.log 2>&1 &

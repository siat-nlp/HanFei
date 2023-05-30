#!/bin/bash


# Main run.
python tasks/task/train.py \
    --experiment_name=bloomz-7b1-mt-sft-gpu8-1e5 \
    --training_stage=instruction_tuning \
    --model_name=gpt  \
    --model_name_or_path=/model_name_or_path \
    --tokenizer_name=/model_name_or_path \
    --data_dir=/data_path \
    --train_batch_size=1 \
    --eval_batch_size=1 \
    --accum_batches_args=16 \
    --max_source_length=4096 \
    --max_target_length=4096 \
    --num_sanity_val_steps=0 \
    --val_check_interval=0.1 \
    --log_every_n_steps=10 \
    --save_every_n_epochs=1 \
    --save_top_k=-1 \
    --max_epochs=3 \
    --max_steps=-1 \
    --lr_scheduler=cosine \
    --learning_rate=1e-5 \
    --warmup_steps=200 \
    --weight_decay=0. \
    --logger_name=WandbLogger \
    --num_workers=8

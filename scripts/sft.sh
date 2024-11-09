#!/bin/bash

MODEL_VERSION=vicuna-v1-5-7b
gpu_vis=2 # per_device_train_batch_size * gradient_accumulation_steps * n_gpus = 128
MASTER_PORT=29571


deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT vtimellm/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --training_stage 3 \
    --model_name_or_path ./checkpoints/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./data_pre/new_stage3_v3.json \
    --feat_folder ./feat/stage3_clip_feat \
    --pretrain_mm_mlp_adapter ./checkpoints/sft_ckp/vtimellm-$MODEL_VERSION-stage1/mm_projector.bin \
    --stage2_path ./checkpoints/sft_ckp/vtimellm-$MODEL_VERSION-stage2 \
    --output_dir ./checkpoints/sft_ckp/vtimellm-$MODEL_VERSION-stage_sft \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --freeze_mm_mlp_adapter True \
    --lora_r 64 \
    --lora_alpha 128 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb 


python ./vtimellm/val/val_charades_ori.py \
    --stage3 checkpoints/vtimellm-vicuna-v1-5-7b-stage_sft \
    --device 2




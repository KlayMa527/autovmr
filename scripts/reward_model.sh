#!/bin/bash

MODEL_VERSION=vicuna-v1-5-7b
gpu_vis=3 # per_device_train_batch_size * gradient_accumulation_steps * n_gpus = 128
MASTER_PORT=29575


deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT vtimellm/reward_model/reward_model_train.py \
    --deepspeed ./scripts/zero2.json


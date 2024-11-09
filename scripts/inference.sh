#!/bin/bash

python ./vtimellm/inference.py \
    --clip_path  ./checkpoints/clip/ViT-L-14.pt  \
    --model_base ./checkpoints/vicuna-7b-v1.5 \
    --pretrain_mm_mlp_adapter  ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2  ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2  \
    --stage3  ./checkpoints/vtimellm-vicuna-v1-5-7b-stage_sft  \
    --video_path  ./images/baby.mp4  \

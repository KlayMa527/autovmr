#! /bin/bash

device=2
exp_name=ppo
epochs=2
num_return_sequences=1
num_return_sequences_span=2
times=v_uniform
ckp=checkpoint-5060
reward_model=./checkpoints/reward_model_ckp/


python   ./vtimellm/ppo/ppo.py \
--stage_2 ./checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage2/ \
--stage_3 ./checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage_sft/ \
--feat_folder ./feat/intern_clip_feat/  \
--reward_model $reward_model  \
--output_dir ./output/ppo/$exp_name/ \
--logging_dir ./output/ppo/$exp_name/ \
--device $device \
--epochs  $epochs \
--data_path ./dataset/ppo_data/ppo.json \
--kl_penalty kl \
--num_return_sequences $num_return_sequences \
--num_return_sequences_span $num_return_sequences_span \
--ppo_epochs  1 \
--generate_batch_size 1 \
--batch_size 128 \
--mini_batch_size 1 > ./output/logs/ppo/$exp_name-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1




python ./vtimellm/val/val_new.py  \
    --stage3  ./checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage_sft  \
    --task ac \
    --ppo_stage  ./output/ppo/$exp_name/epoch0_final  \
    --device $device \
    --ppo   > ./output/logs/val/$exp_name-ac-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



python  ./vtimellm/val/val_new.py  \
    --stage3  ./checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage_sft \
    --task sta \
    --ppo_stage   ./output/ppo/$exp_name/epoch0_final  \
    --device $device \
    --ppo    > ./output/logs/val/$exp_name-sta-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1

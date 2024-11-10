#! /bin/bash

device=3
exp_name=lsv_l_1_2
epochs=1
num_return_sequences=1
num_return_sequences_span=2
times=v_uniform
ckp=checkpoint-5060
reward_model=./output/reward_model/from_v5/v_uniform/checkpoint-5060/


python   ./vtimellm/ppo/ppo_new.py \
--stage_2 ./checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage2/ \
--stage_3 ./checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage3_5/ \
--feat_folder ./feat/intern_clip_feat/  \
--reward_model $reward_model  \
--output_dir ./output/lsv/$exp_name/ \
--logging_dir ./output/lsv/$exp_name/ \
--device $device \
--epochs  $epochs \
--data_path ./dataset/lsv_data/lvd.json \
--kl_penalty kl \
--num_return_sequences $num_return_sequences \
--num_return_sequences_span $num_return_sequences_span \
--ppo_epochs  1 \
--generate_batch_size 1 \
--batch_size 128 \
--mini_batch_size 1 > ./output/logs/lsv/$exp_name-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1




python ./vtimellm/val/val_new.py  \
    --stage3  ./checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task ac \
    --ppo_stage  ./output/lsv/$exp_name/epoch0_final  \
    --device $device \
    --ppo   > ./output/logs/val/$exp_name-ac-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



python  ./vtimellm/val/val_new.py  \
    --stage3  ./checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task sta \
    --ppo_stage   ./output/lsv/$exp_name/epoch0_final  \
    --device $device \
    --ppo    > ./output/logs/val/$exp_name-sta-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



# python  ./vtimellm/val/val_new.py  \
#     --stage3  ./checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage3_5  \
#     --task ac \
#     --ppo_stage  ./output/lsv/$exp_name/epoch1_final  \
#     --device $device \
#     --ppo   > ./output/logs/val/$exp_name-ac-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



# python  ./vtimellm/val/val_new.py  \
#     --stage3  ./checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage3_5  \
#     --task sta \
#     --ppo_stage   ./output/lsv/$exp_name/epoch1_final  \
#     --device $device \
#     --ppo    > ./output/logs/val/$exp_name-sta-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1







# python ./vtimellm/val/val_new.py  \
#     --stage3  ./checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage3_5  \
#     --task ac \
#     --ppo_stage  ./output/lsv/$exp_name/epoch2_final  \
#     --device $device \
#     --ppo   > ./output/logs/val/$exp_name-ac-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



# python  ./vtimellm/val/val_new.py  \
#     --stage3  ./checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage3_5  \
#     --task sta \
#     --ppo_stage   ./output/lsv/$exp_name/epoch2_final  \
#     --device $device \
#     --ppo    > ./output/logs/val/$exp_name-sta-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



# python  ./vtimellm/val/val_new.py  \
#     --stage3  ./checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage3_5  \
#     --task ac \
#     --ppo_stage  ./output/lsv/$exp_name/epoch3_final  \
#     --device $device \
#     --ppo   > ./output/logs/val/$exp_name-ac-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



# python  ./vtimellm/val/val_new.py  \
#     --stage3  ./checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage3_5  \
#     --task sta \
#     --ppo_stage   ./output/lsv/$exp_name/epoch3_final  \
#     --device $device \
#     --ppo    > ./output/logs/val/$exp_name-sta-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1




# 0 19 * * * bash ./vtimellm/combine/combine.sh



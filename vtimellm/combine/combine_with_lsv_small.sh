#! /bin/bash

device=1
exp_name=combine_with_lsv_small_1_2
epochs=2
num_return_sequences=1
num_return_sequences_span=2
times=v_uniform
ckp=checkpoint-5060
reward_model=./output/reward_model/from_v5/v_uniform/checkpoint-5060/


python   ./vtimellm/ppo/ppo.py \
--stage_2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2/ \
--stage_3 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5/ \
--feat_folder ./dataset/combine_with_lsv_data/feat/  \
--reward_model $reward_model  \
--output_dir ./output/combine/$exp_name/ \
--logging_dir ./output/combine/$exp_name/ \
--device $device \
--epochs  $epochs \
--data_path  ./dataset/combine_with_lsv_data/combine_with_lsv_data.json \
--kl_penalty kl \
--num_return_sequences $num_return_sequences \
--num_return_sequences_span $num_return_sequences_span \
--ppo_epochs  1 \
--generate_batch_size 1 \
--batch_size 128 \
--mini_batch_size 1 > ./output/logs/combine/$exp_name-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1


python ./vtimellm/val/val_new.py  \
    --stage3  ./checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task ac \
    --ppo_stage  ./output/combine/$exp_name/epoch0_final  \
    --device $device \
    --ppo   > ./output/logs/val/$exp_name-ac-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



python  ./vtimellm/val/val_new.py  \
    --stage3  ./checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task sta \
    --ppo_stage   ./output/combine/$exp_name/epoch0_final  \
    --device $device \
    --ppo    > ./output/logs/val/$exp_name-sta-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



python  ./vtimellm/val/val_new.py  \
    --stage3  ./checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task ac \
    --ppo_stage  ./output/combine/$exp_name/epoch1_final  \
    --device $device \
    --ppo   > ./output/logs/val/$exp_name-ac-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



python  ./vtimellm/val/val_new.py  \
    --stage3  ./checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task sta \
    --ppo_stage   ./output/combine/$exp_name/epoch1_final  \
    --device $device \
    --ppo    > ./output/logs/val/$exp_name-sta-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1




# 0 19 * * * bash ./vtimellm/combine/combine.sh








# python   ./vtimellm/ppo/ppo_new.py \
# --stage_2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2/ \
# --stage_3 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5/ \
# --feat_folder ./dataset/combine_data/feat/  \
# --reward_model ./output/reward_model/from_v5/v_uniform/checkpoint-5060/ \
# --output_dir ./output/combine/combine_v_2/ \
# --logging_dir ./output/combine/combine_v_2/ \
# --device 1 \
# --epochs  1 \
# --data_path ./dataset/combine_data/combine.json \
# --kl_penalty kl \
# --num_return_sequences 2 \
# --num_return_sequences_span 2 \
# --ppo_epochs 1 \
# --generate_batch_size 1 \
# --batch_size 128 \
# --mini_batch_size 1 > ./output/logs/combine_2_`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



# python ./vtimellm/val/val_activitynet.py \
#     --stage3 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
#     --feat_folder ./dataset/activitynet/feat/val  \
#     --ppo_stage ./output/combine/combine_v_2/epoch0_final  \
#     --log_path ./output/val/from_v5/activity_net/  \
#     --device 1  \
#     --ppo


# python ./vtimellm/val/val_charades_sta.py \
#     --stage3 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
#     --feat_folder ./dataset/charades_sta/feat/  \
#     --ppo_stage ./output/combine/combine_v_2/epoch0_final  \
#     --log_path ./output/val/from_v5/charades_sta/  \
#     --device 1  \
#     --ppo

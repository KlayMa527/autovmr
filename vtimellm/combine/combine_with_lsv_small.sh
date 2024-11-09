#! /bin/bash

device=1
exp_name=combine_with_lsv_small_1_2
epochs=2
num_return_sequences=1
num_return_sequences_span=2
times=v_uniform
ckp=checkpoint-5060
reward_model=/home/luoshu/VTimeLLM/output/reward_model/from_v5/v_uniform/checkpoint-5060/


python   /home/luoshu/VTimeLLM/vtimellm/ppo/ppo.py \
--stage_2 /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage2/ \
--stage_3 /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5/ \
--feat_folder /home/luoshu/VTimeLLM/dataset/combine_with_lsv_data/feat/  \
--reward_model $reward_model  \
--output_dir /home/luoshu/VTimeLLM/output/combine/$exp_name/ \
--logging_dir /home/luoshu/VTimeLLM/output/combine/$exp_name/ \
--device $device \
--epochs  $epochs \
--data_path  /home/luoshu/VTimeLLM/dataset/combine_with_lsv_data/combine_with_lsv_data.json \
--kl_penalty kl \
--num_return_sequences $num_return_sequences \
--num_return_sequences_span $num_return_sequences_span \
--ppo_epochs  1 \
--generate_batch_size 1 \
--batch_size 128 \
--mini_batch_size 1 > /home/luoshu/VTimeLLM/output/logs/combine/$exp_name-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1


python /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
    --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task ac \
    --ppo_stage  /home/luoshu/VTimeLLM/output/combine/$exp_name/epoch0_final  \
    --device $device \
    --ppo   > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-ac-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
    --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task sta \
    --ppo_stage   /home/luoshu/VTimeLLM/output/combine/$exp_name/epoch0_final  \
    --device $device \
    --ppo    > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-sta-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
    --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task ac \
    --ppo_stage  /home/luoshu/VTimeLLM/output/combine/$exp_name/epoch1_final  \
    --device $device \
    --ppo   > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-ac-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
    --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task sta \
    --ppo_stage   /home/luoshu/VTimeLLM/output/combine/$exp_name/epoch1_final  \
    --device $device \
    --ppo    > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-sta-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1




# 0 19 * * * bash /home/luoshu/VTimeLLM/vtimellm/combine/combine.sh








# python   /home/luoshu/VTimeLLM/vtimellm/ppo/ppo_new.py \
# --stage_2 /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage2/ \
# --stage_3 /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5/ \
# --feat_folder /home/luoshu/VTimeLLM/dataset/combine_data/feat/  \
# --reward_model /home/luoshu/VTimeLLM/output/reward_model/from_v5/v_uniform/checkpoint-5060/ \
# --output_dir /home/luoshu/VTimeLLM/output/combine/combine_v_2/ \
# --logging_dir /home/luoshu/VTimeLLM/output/combine/combine_v_2/ \
# --device 1 \
# --epochs  1 \
# --data_path /home/luoshu/VTimeLLM/dataset/combine_data/combine.json \
# --kl_penalty kl \
# --num_return_sequences 2 \
# --num_return_sequences_span 2 \
# --ppo_epochs 1 \
# --generate_batch_size 1 \
# --batch_size 128 \
# --mini_batch_size 1 > /home/luoshu/VTimeLLM/output/logs/combine_2_`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



# python /home/luoshu/VTimeLLM/vtimellm/val/val_activitynet.py \
#     --stage3 /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
#     --feat_folder /home/luoshu/VTimeLLM/dataset/activitynet/feat/val  \
#     --ppo_stage /home/luoshu/VTimeLLM/output/combine/combine_v_2/epoch0_final  \
#     --log_path /home/luoshu/VTimeLLM/output/val/from_v5/activity_net/  \
#     --device 1  \
#     --ppo


# python /home/luoshu/VTimeLLM/vtimellm/val/val_charades_sta.py \
#     --stage3 /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
#     --feat_folder /home/luoshu/VTimeLLM/dataset/charades_sta/feat/  \
#     --ppo_stage /home/luoshu/VTimeLLM/output/combine/combine_v_2/epoch0_final  \
#     --log_path /home/luoshu/VTimeLLM/output/val/from_v5/charades_sta/  \
#     --device 1  \
#     --ppo

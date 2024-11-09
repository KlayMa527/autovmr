#! /bin/bash

device=2
exp_name=lsv_s_uni_1_2_v2
epochs=4
num_return_sequences=1
num_return_sequences_span=2
times=v_uniform
ckp=checkpoint-5060
reward_model=/home/luoshu/VTimeLLM/output/reward_model/from_v5/v_uniform/checkpoint-5060/


/home/luoshu/miniconda3/envs/vtimellm/bin/python   /home/luoshu/VTimeLLM/vtimellm/ppo/ppo_new.py \
--stage_2 /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage2/ \
--stage_3 /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5/ \
--feat_folder /home/luoshu/VTimeLLM/feat/intern_clip_feat/  \
--reward_model $reward_model  \
--output_dir /home/luoshu/VTimeLLM/output/lsv/$exp_name/ \
--logging_dir /home/luoshu/VTimeLLM/output/lsv/$exp_name/ \
--device $device \
--epochs  $epochs \
--data_path /home/luoshu/VTimeLLM/dataset/lsv_data/lvd_small.json \
--kl_penalty kl \
--num_return_sequences $num_return_sequences \
--num_return_sequences_span $num_return_sequences_span \
--ppo_epochs  1 \
--generate_batch_size 1 \
--batch_size 128 \
--mini_batch_size 1 > /home/luoshu/VTimeLLM/output/logs/lsv/$exp_name-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1




/home/luoshu/miniconda3/envs/vtimellm/bin/python /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
    --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task ac \
    --ppo_stage  /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch0_final  \
    --device $device \
    --ppo   > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-ac-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



/home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
    --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task sta \
    --ppo_stage   /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch0_final  \
    --device $device \
    --ppo    > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-sta-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



/home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
    --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task ac \
    --ppo_stage  /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch1_final  \
    --device $device \
    --ppo   > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-ac-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



/home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
    --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task sta \
    --ppo_stage   /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch1_final  \
    --device $device \
    --ppo    > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-sta-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1







/home/luoshu/miniconda3/envs/vtimellm/bin/python /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
    --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task ac \
    --ppo_stage  /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch2_final  \
    --device $device \
    --ppo   > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-ac-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



/home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
    --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task sta \
    --ppo_stage   /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch2_final  \
    --device $device \
    --ppo    > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-sta-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



/home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
    --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task ac \
    --ppo_stage  /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch3_final  \
    --device $device \
    --ppo   > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-ac-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



/home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
    --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task sta \
    --ppo_stage   /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch3_final  \
    --device $device \
    --ppo    > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-sta-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1




# 0 19 * * * bash /home/luoshu/VTimeLLM/vtimellm/combine/combine.sh



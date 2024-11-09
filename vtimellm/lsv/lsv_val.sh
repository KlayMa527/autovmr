device=1
exp_name=lsv_l_1_2

/home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
    --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task ac \
    --ppo_stage  /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch0_550  \
    --device $device \
    --ppo   > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-ac-550-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1

/home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
    --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task ac \
    --ppo_stage  /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch0_500  \
    --device $device \
    --ppo   > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-ac-500-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1


/home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
    --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task ac \
    --ppo_stage  /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch0_400  \
    --device $device \
    --ppo   > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-ac-400-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1

/home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
    --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task ac \
    --ppo_stage  /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch0_300  \
    --device $device \
    --ppo   > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-ac-300-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1



/home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
    --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task ac \
    --ppo_stage  /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch0_200  \
    --device $device \
    --ppo   > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-ac-200-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1


/home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
    --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task ac \
    --ppo_stage  /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch0_100  \
    --device $device \
    --ppo   > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-ac-100-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1






# /home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
#     --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
#     --task sta \
#     --ppo_stage   /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch0_550  \
#     --device $device \
#     --ppo    > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-sta`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1





# /home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
#     --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
#     --task ac \
#     --ppo_stage  /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch0_100  \
#     --device $device \
#     --ppo   > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-ac`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1

# /home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
#     --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
#     --task sta \
#     --ppo_stage   /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch0_100  \
#     --device $device \
#     --ppo    > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-sta`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1

# /home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
#     --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
#     --task ac \
#     --ppo_stage  /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch0_200  \
#     --device $device \
#     --ppo   > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-ac`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1

# /home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
#     --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
#     --task sta \
#     --ppo_stage   /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch0_200  \
#     --device $device \
#     --ppo    > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-sta`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1


# /home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
#     --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
#     --task ac \
#     --ppo_stage  /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch0_300  \
#     --device $device \
#     --ppo   > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-ac`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1

# /home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
#     --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
#     --task sta \
#     --ppo_stage   /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch0_300  \
#     --device $device \
#     --ppo    > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-sta`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1


# /home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
#     --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
#     --task ac \
#     --ppo_stage  /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch0_400  \
#     --device $device \
#     --ppo   > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-ac`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1

# /home/luoshu/miniconda3/envs/vtimellm/bin/python  /home/luoshu/VTimeLLM/vtimellm/val/val_new.py  \
#     --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
#     --task sta \
#     --ppo_stage   /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch0_400  \
#     --device $device \
#     --ppo    > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-sta`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1
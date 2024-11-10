device=1
exp_name=lsv_l_1_2

/home/luoshu/miniconda3/envs/vtimellm/bin/python  ./vtimellm/val/val_new.py  \
    --stage3  ./checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task ac \
    --ppo_stage  ./output/lsv/$exp_name/epoch0_550  \
    --device $device \
    --ppo   > ./output/logs/val/$exp_name-ac-550-`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1


/home/luoshu/miniconda3/envs/vtimellm/bin/python  ./vtimellm/val/val_new.py  \
    --stage3  /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --task sta \
    --ppo_stage   /home/luoshu/VTimeLLM/output/lsv/$exp_name/epoch0_300  \
    --device $device \
    --ppo    > /home/luoshu/VTimeLLM/output/logs/val/$exp_name-sta`date +\%Y\%m\%d\%H\%M\%S`_ppo_run.log 2>&1
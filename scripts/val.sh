python /home/luoshu/VTimeLLM/vtimellm/val/val_activitynet.py \
    --stage3 /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --feat_folder /home/luoshu/VTimeLLM/dataset/activitynet/feat/val  \
    --ppo_stage  /home/luoshu/VTimeLLM/output/ppo/ppo_fromv5_exp_8/epoch0_final  \
    --log_path /home/luoshu/VTimeLLM/output/val/from_v5/activity_net/  \
    --device 3  \
    --ppo



python /home/luoshu/VTimeLLM/vtimellm/val/val_charades_sta.py \
    --stage3 /home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --feat_folder /home/luoshu/VTimeLLM/dataset/charades_sta/feat/  \
    --ppo_stage  /home/luoshu/VTimeLLM/output/ppo/ppo_fromv5_exp_8/epoch0_final  \
    --log_path /home/luoshu/VTimeLLM/output/val/from_v5/charades_sta/  \
    --device 3  \
    --ppo

python ./vtimellm/val/val_activitynet.py \
    --stage3 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --feat_folder ./dataset/activitynet/feat/val  \
    --ppo_stage  ./output/ppo/ppo_fromv5_exp_8/epoch0_final  \
    --log_path ./output/val/from_v5/activity_net/  \
    --device 0  \
    --ppo



python ./vtimellm/val/val_charades_sta.py \
    --stage3 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5  \
    --feat_folder ./dataset/charades_sta/feat/  \
    --ppo_stage  ./output/ppo/ppo_fromv5_exp_8/epoch0_final  \
    --log_path ./output/val/from_v5/charades_sta/  \
    --device 0  \
    --ppo

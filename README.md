# AutoVMR:An Autonomous Event Generation and Localization Approach for Video Moment Retrieval

## experiment result
  experiment is stored in ./output


## Installation 

### a. Requirements
```shell
pip install -r requirements.txt
```
### b. Datasets

* Download dataset from [BaiduYun(vj2d)](https://pan.baidu.com/s/1D1iy7OY1fOfJEO062PP7_A). Place them into the 'feat' directory.

### c. Pretrain weight and checkpoint
AutoVMR needs to go through three stages: SFT, Reward model training, and PPO training. Please follow the instructions below to train AutoVMR-7B model.
* Download [clip](https://cloud.tsinghua.edu.cn/d/6db5d02883124826aa6f/?p=%2Fcheckpoints&mode=list) and [Vicuna v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) weights, and place them into the 'checkpoints' directory.
* Download [sft checkpoint(901c)](https://pan.baidu.com/s/1sqXM1iCM6Zan4ixvEWyaog) weight, and place them into the 'checkpoints/sft_ckp/'
* Download [reward model checkpoint(fcl1)](https://pan.baidu.com/s/1-zF_OOewQKjvnkQinrnFmA) weight, and place them into the 'checkpoints/reward_model_ckp/'
* Download [ppo checkpoint(m0hw)](https://pan.baidu.com/s/1sqXM1iCM6Zan4ixvEWyaog) weight, and place them into the 'checkpoints/ppo_ckp/'

```shell
--  checkpoints
      --  clip
      --  ppo_ckp

      --  reward_model_ckp
          --  v_uniform
      --  sft_ckp
          --  vtimellm-vicuna-v1-5-7b-stage1
          --  vtimellm-vicuna-v1-5-7b-stage2
          --  vtimellm-vicuna-v1-5-7b-stage_sft
      --  vicuna-7b-v1.5
--  feat

--  scripts
      -- sft.sh
      -- reward_model.sh
      -- ppo.sh
      -- inference.sh

```


## Getting Started

###  a. sft training
```
bash scripts/ppo.sh
```
###  b. ppo training
```
bash scripts/ppo.sh
```

### c. val
```
bash scripts/val.sh
```


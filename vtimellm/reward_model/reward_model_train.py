import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(root_dir)

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Tuple, Union
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from trl import RewardConfig, RewardTrainer, is_xpu_available
import torch
import torch.nn as nn
from peft import PeftModel, get_peft_model
from transformers import PreTrainedTokenizerBase
import json
import pathlib
from vtimellm.model.vtimellm_llama_reward import VTimeLLMLlamaForSequenceClassification
from vtimellm.train.dataset import RewardModelDataset, DataCollatorForRewardDataet
from vtimellm.reward_model.reward_model_trainer import VtimeLlmRewardTrainer

tqdm.pandas()


@dataclass
class ScriptArguments:
    model_name: str = "./checkpoints/vicuna-7b-v1.5/"
    stage_2_path: str = './checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage2/'
    stage_3_path: str = './checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage_sft/'
    '''load data precision'''
    load_in_8bit: bool = False
    """load the model in 8 bits precision"""
    load_in_4bit: bool = False
    """load the model in 4 bits precision"""
    trust_remote_code: bool = True
    """Enable `trust_remote_code`"""
    '''reward model config'''
    reward_config: RewardConfig = field(default_factory=lambda: RewardConfig(
        output_dir="./output/reward_model_ckp/",
        per_device_train_batch_size=256,
        num_train_epochs=20,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=1.41e-4,
        report_to="tensorboard",
        remove_unused_columns=False,
        optim="adamw_torch",
        logging_steps=1,
        evaluation_strategy="epoch",
        max_length=2048,
        save_strategy='epoch',
        lr_scheduler_type = 'cosine'
    ))

    use_peft: bool = True
    """whether to use peft"""

    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            # task_type="SEQ_CLS",
            modules_to_save=["scores"],
        ), )

    pretrain_mm_mlp_adapter: str = './checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin'
    freeze_mm_mlp_adapter: bool = True
    # max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})


@dataclass
class DataArguments:
    train_data_path: str = './dataset/reward_model_data/reward_model_train_v_unif.json'
    eval_data_path: str = './dataset/reward_model_data/reward_model_val_v_unif.json'
    feat_folder: str = './feat/stage3_clip_feat/'



script_args = ScriptArguments()
# script_args.reward_config.evaluation_strategy = "steps" if script_args.eval_split != "none" else "no"

dataset_args = DataArguments()

# torch.cuda.set_device(script_args.device)
# define load lora function
def load_lora(model, lora_path):
    non_lora_trainables_path = os.path.join(lora_path,
                                            'non_lora_trainables.bin')
    if os.path.exists(non_lora_trainables_path):
        non_lora_trainables = torch.load(non_lora_trainables_path,
                                         map_location='cpu')
        non_lora_trainables = {
            (k[11:] if k.startswith('base_model.') else k): v
            for k, v in non_lora_trainables.items()
        }
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {
                (k[6:] if k.startswith('model.') else k): v
                for k, v in non_lora_trainables.items()
            }
        model.load_state_dict(non_lora_trainables, strict=False)
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, lora_path)
    return model


# define find_all_linear_names
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if 'score' not in name:
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) ==1 else names[-1])
    print(lora_module_names)
    return list(lora_module_names)


'''
step 1 define model and load lora module
'''
# Step 1: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError(
        "You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit,
        load_in_4bit=script_args.load_in_4bit)
    # Copy the model to each device
    device_map = ({
        "": f"xpu:{Accelerator().local_process_index}"
    } if is_xpu_available() else {
        "": Accelerator().local_process_index
    })
else:
    device_map = None
    quantization_config = None

# load model
model = VTimeLLMLlamaForSequenceClassification.from_pretrained(
    script_args.model_name, num_labels=1)
model.to(torch.bfloat16)

# save origin module

#set loar_config
lora_config = LoraConfig(r=64,
                         lora_alpha=16,
                         target_modules=find_all_linear_names(model),
                         bias="none",
                         lora_dropout=0.05,
                         task_type="SEQ_CLS")

# merge lora
# load stage_2 lora

print('Loading stage2 weights...')
model = load_lora(model, script_args.stage_2_path)
print('Merging stage2 weights...')
model = model.merge_and_unload()

# load stage_3 lora
print('Loading stage3 weights...')
model = load_lora(model, script_args.stage_3_path)
print('Merging stage3 weights...')
model = model.merge_and_unload()

# add adapter
print("Adding LoRA adapters...")
model = get_peft_model(model, lora_config)

# define mm_projector_weight
model.get_model().initialize_vision_modules(model_args=script_args)
model.get_model().mm_projector.to(torch.float16)

if script_args.freeze_mm_mlp_adapter:
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

# for name, param in model.get_model().named_parameters():
#         # 假设 LoRA 参数的名称中包含 'lora'
#         param.requires_grad = False

# for name, param in model.named_parameters():
#         if 'score' in name and not 'lora' in name:
#             param.requires_grad = False

# define tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
'''
step 2 define dataset
'''
# define dataset
train_dataset = RewardModelDataset(tokenizer=tokenizer,
                                   data_path=dataset_args.train_data_path,
                                   data_args=dataset_args)

# define data collector
data_collator = DataCollatorForRewardDataet(tokenizer=tokenizer)
# defien eval dataset
eval_dataset = RewardModelDataset(tokenizer=tokenizer,
                                  data_path=dataset_args.eval_data_path,
                                  data_args=dataset_args)

'''
step 3 define trainer and train
'''
# reward_config = script_args.reward_config
# print(reward_config.train_batch_size)
# print(reward_config.per_device_train_batch_size)
# define trainer
trainer = VtimeLlmRewardTrainer(model=model,
                                tokenizer=tokenizer,
                                args=script_args.reward_config,
                                train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                                data_collator=data_collator)
if list(
        pathlib.Path(
            script_args.reward_config.output_dir).glob("checkpoint-*")):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()
trainer.save_model(script_args.reward_config.output_dir)

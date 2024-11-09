import os
import sys
print(1)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(root_dir)
import torch
import torch.nn as nn
from peft import PeftModel
from dataclasses import dataclass,field

from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from tqdm import tqdm
from vtimellm.model.vtimellm_llama_reward import VTimeLLMLlamaForSequenceClassification
from vtimellm.train.dataset import RewardModelDataset, DataCollatorForRewardDataet
from vtimellm.reward_model.reward_model_trainer import VtimeLlmRewardTrainer
from trl import RewardConfig

tqdm.pandas()

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

@dataclass
class ScriptArguments:
    model_name: str = "./checkpoints/vicuna-7b-v1.5/"
    stage_2_path: str = './checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage2/'
    stage_3_path: str = './checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage_sft/'
    reward_model_path = './output/reward_model_val/'
    use_peft: bool = True
    pretrain_mm_mlp_adapter: str = './checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin'
    freeze_mm_mlp_adapter: bool = True

    reward_config: RewardConfig = field(default_factory=lambda: RewardConfig(
        output_dir="./output/val",
        per_device_train_batch_size=256,
        num_train_epochs=20,
        gradient_accumulation_steps=2,
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
    ))


@dataclass
class DataArguments:
    eval_data_path: str = './dataset/reward_model_data/reward_model_val_v_unif.json'
    feat_folder: str = './feat/stage3_clip_feat/'


script_args = ScriptArguments()

dataset_args = DataArguments()

testing_args = TrainingArguments(
    output_dir='./output/reward_val')

model = VTimeLLMLlamaForSequenceClassification.from_pretrained(
    script_args.model_name, num_labels=1)




model.to(torch.bfloat16)

print('Loading stage2 weights...')
model = load_lora(model, script_args.stage_2_path)
print('Merging stage2 weights...')
model = model.merge_and_unload()

# load stage_3 lora
print('Loading stage3 weights...')
model = load_lora(model, script_args.stage_3_path)
print('Merging stage3 weights...')
model = model.merge_and_unload()

# load reward model
print('Loading reward model...')
model = load_lora(model, script_args.reward_model_path)
print('Merging reward model weights...')
model = model.merge_and_unload()



# define mm_projector_weight
model.get_model().initialize_vision_modules(model_args=script_args)
model.get_model().mm_projector.to(torch.float16)

if script_args.freeze_mm_mlp_adapter:
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

# define tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

# define test dataset

eval_dataset = RewardModelDataset(tokenizer=tokenizer,
                                  data_path=dataset_args.eval_data_path,
                                  data_args=dataset_args)
data_collator = DataCollatorForRewardDataet(tokenizer=tokenizer)

tester = VtimeLlmRewardTrainer(model=model,
                               tokenizer=tokenizer,
                               args=script_args.reward_config,
                               eval_dataset=eval_dataset,
                               data_collator=data_collator)
output = tester.predict(test_dataset=eval_dataset)
print(output)



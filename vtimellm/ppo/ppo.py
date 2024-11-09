import os
import sys
import argparse
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(root_dir)

from dataclasses import dataclass, field
from typing import Optional
import random
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available

from vtimellm.train.dataset import PPOModelDataset, DataCollatorForPPODataset

from peft import PeftModel, get_peft_model

from vtimellm.ppo.vtimellm_ppo_trainer import VtimeLLMPPOTrainer
from vtimellm.mm_utils import tokenizer_image_token
from vtimellm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
import re
from vtimellm.ppo.ppo_util import generate_time_span_query, find_all_linear_names, generate_reward_model_and_forward_input, load_reward_model, load_pretrained_model
import json



tqdm.pandas()


def ppo_train(ppo_trainer, query, response, score, images, total_batch_num,
              batch_size):
    data_num = len(query)
    for batch_start in range(0, data_num, batch_size):
        end = batch_start + batch_size
        batch_end = end if end < data_num else data_num
        ppo_query = query[batch_start:batch_end]
        ppo_response = response[batch_start:batch_end]
        ppo_score = score[batch_start:batch_end]
        ppo_images = images[batch_start:batch_end]
        batch_len = batch_end - batch_start
        if batch_len <  batch_size:
            for i in range(batch_size - batch_len):
                index = random.randint(0, data_num-1)
                print('data_num:', data_num)
                print('index:', index)
                ppo_query.append(query[index])
                ppo_response.append(response[index])
                ppo_score.append(score[index])
                ppo_images = torch.cat((ppo_images, images[index:index + 1]))
        stats = ppo_trainer.step(ppo_query,
                                 ppo_response,
                                 ppo_score,
                                 images=ppo_images)
        ppo_trainer.log_stats(stats, batch,
                              [item.to(torch.float16) for item in score])



@dataclass
class ScriptArgument:
    clip_path: str = './checkpoints/clip/ViT-L-14.pt'
    model_base: str = "./checkpoints/vicuna-7b-v1.5/"
    pretrain_mm_mlp_adapter: str = './checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin'
    freeze_mm_mlp_adapter: bool = True
    stage_2: str = './checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage2/'
    stage_3: str = './checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage_sft/'
    reward_model: str = './output/reward_model_ckp/'
    output_dir: str = './output/ppo/ppo_exp/'
    epochs: int = 5
    use_seq2seq: bool = False
    use_peft: bool = True
    device: int = 1
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Enable `trust_remote_code`"})


@dataclass
class DataArguments:
    data_path: str = './dataset/sft_data/sft_data.json'
    feat_folder: str = './feat/stage3_clip_feat/'
    query: str = '<video>\nCan you provide a detailed introduction to the events that occurred in the video and provide corresponding time periods?'
    reward_model_prompt: str = '<video>\nCan you provide a detailed introduction to the events that occurred in the video?'


def parse_args():
    parser = argparse.ArgumentParser(description='PPO Trainer.')
    parser.add_argument(
        '--stage_2',
        type=str,
        default=
        './checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage2/')
    parser.add_argument(
        '--stage_3',
        type=str,
        default=
        './checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage_sft/')
    parser.add_argument(
        '--reward_model',
        type=str,
        default=
        './checkpoints/reward_model_ckp/'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output/ppo/ppo_exp/')
    parser.add_argument(
        '--logging_dir',
        type=str,
        default='./output/ppo/ppo_exp/')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument(
        '--data_path',
        type=str,
        default='./dataset/sft_data/sft_data.json')
    parser.add_argument('--kl_penalty', type=str, default='full')
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--num_return_sequences_span', type=int, default=1)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--mini_batch_size', type=int, default=4)
    parser.add_argument('--generate_batch_size', type=int, default=2)
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    script_args = ScriptArgument()
    dataset_args = DataArguments()
    args = parse_args()
    script_args.stage_2 = args.stage_2
    script_args.stage_3 = args.stage_3
    script_args.reward_model = args.reward_model
    script_args.output_dir = args.output_dir
    script_args.device = args.device
    dataset_args.data_path = args.data_path
    generate_batch_size = args.generate_batch_size

    # write Hyperparameter
    os.makedirs(args.output_dir, exist_ok=True)
    hyp_path = os.path.join(args.output_dir, 'hyp.json')
    args_dict = vars(args)
    with open(hyp_path, 'w') as hyp_file:
        json.dump(args_dict, hyp_file, indent=2)

    ppo_config = PPOConfig(learning_rate=1.41e-5,
                           log_with="tensorboard",
                           mini_batch_size=args.mini_batch_size,
                           batch_size=args.batch_size,
                           ppo_epochs=args.ppo_epochs,
                           gradient_accumulation_steps=int(
                               args.batch_size / args.mini_batch_size),
                           early_stopping=False,
                           target_kl=6,
                           kl_penalty=args.kl_penalty,
                           seed=0,
                           use_score_scaling=False,
                           use_score_norm=False,
                           score_clip=None,
                           project_kwargs={"logging_dir": args.logging_dir})

    print(args)
    print(script_args)
    print(dataset_args)
    print(ppo_config)
    # define device
    torch.cuda.set_device(script_args.device)

    # load pretrain model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_base)
    model = load_pretrained_model(script_args, script_args.stage_2,
                                  script_args.stage_3)

    #define lora config
    lora_config = LoraConfig(r=64,
                             lora_alpha=16,
                             target_modules=find_all_linear_names(model),
                             bias="none",
                             lora_dropout=0.05,
                             task_type="CAUSAL_LM")

    # define model
    model = get_peft_model(model, lora_config)

    model.get_model().initialize_vision_modules(script_args)
    model.get_model().mm_projector.to(torch.float16)
    if script_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    model = model.cuda()
    # model.to(torch.bfloat16)

    # define ref_model
    # _, ref_model, __ = load_pretrained_model(script_args, script_args.stage_2, script_args.stage_3)
    # ref_model = ref_model.cuda()
    # ref_model.to(torch.float16)

    # define dataset and data collator
    ppo_dataset = PPOModelDataset(query=dataset_args.query,
                                  tokenizer=tokenizer,
                                  data_path=dataset_args.data_path,
                                  data_args=dataset_args)
    data_collator = DataCollatorForPPODataset(tokenizer=tokenizer)


    # define PPO Trainer
    ppo_trainer = VtimeLLMPPOTrainer(ppo_config,
                                     model=model,
                                     ref_model=None,
                                     tokenizer=tokenizer,
                                     dataset=ppo_dataset,
                                     data_collator=data_collator)

    # load reward model
    reward_model = load_reward_model(script_args, script_args.stage_2,
                                     script_args.stage_3,
                                     script_args.reward_model)
    reward_model.cuda()
    # reward_model.to(torch.bfloat16)

    # define generate parameters
    num_return_sequences = args.num_return_sequences
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "num_beams": 1,
        # "max_new_tokens": 32,
        'num_return_sequences': num_return_sequences
    }
    num_return_sequences_span = args.num_return_sequences_span
    generation_kwargs_span = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        # "temperature": 0.05,
        "pad_token_id": tokenizer.eos_token_id,
        "num_beams": 1,
        # "max_new_tokens": 32,
        'num_return_sequences': num_return_sequences_span
    }
    data_loader = ppo_trainer.dataloader
    bs = ppo_trainer.config.batch_size
    mini_bs = ppo_trainer.config.mini_batch_size
    # make train loop
    for epoch in range(script_args.epochs):
        for index, batch in tqdm(enumerate(data_loader),
                                 total=len(data_loader),
                                 desc=f"Epoch {epoch+1}/{script_args.epochs}"):
            images = batch['images']
            input_ids = batch['input_ids']
            prompt = batch['prompt']
            # generate query

            responses_tensor = ppo_trainer.generate(
                query_tensor=input_ids,
                images=images,
                return_prompt=False,
                generate_ref_response=False,
                batch_size=generate_batch_size,
                **generation_kwargs)
            responses = tokenizer.batch_decode(responses_tensor)

            # refpiece id is out of range._response = tokenizer.batch_decode(ref_response_tensor)
            images_span, input_ids_span, prompts_span, matches_span = generate_time_span_query(
                images=images,
                responses=responses,
                tokenizer=tokenizer,
                num_return_sequences=num_return_sequences,
                prompt=prompt)

            # generate time span
            with torch.inference_mode():
                responses_tensor_span, ref_responses_tensor_span = ppo_trainer.generate(
                    query_tensor=input_ids_span,
                    images=images_span,
                    return_prompt=False,
                    generate_ref_response=True,
                    batch_size=generate_batch_size,
                    **generation_kwargs_span)
                responses_span = tokenizer.batch_decode(responses_tensor_span)
                ref_responses_span = tokenizer.batch_decode(
                    ref_responses_tensor_span)

            if index % 10 == 0:
                print(responses)
                print(responses_span)
                print(ref_responses_span)

            # generate reward model and  ppo input
            with torch.inference_mode():
                ppo_images, ppo_query, ppo_response, ppo_input_ids, ppo_prompt, ppo_match, reward_model_images, reward_model_input_ids, reward_model_prompt_list = generate_reward_model_and_forward_input(
                    images=images_span,
                    responses=responses_span,
                    tokenizer=tokenizer,
                    num_return_sequences=num_return_sequences_span,
                    prompt=prompts_span,
                    matches=matches_span,
                    reward_model_prompt=dataset_args.reward_model_prompt)

            # torch.cuda.empty_cache()

            # generate score
            with torch.no_grad():
                score = []
                reward_length = reward_model_input_ids.shape[0]
                reward_batch_length = ppo_trainer.config.mini_batch_size
                for batch_start in range(0, reward_length,
                                         reward_batch_length):
                    batch_end = batch_start + reward_batch_length if batch_start + reward_batch_length < reward_length else reward_length
                    input_ids_batch = reward_model_input_ids[
                        batch_start:batch_end, ::]
                    images_batch = reward_model_images[batch_start:batch_end]
                    score_batch = reward_model(input_ids=input_ids_batch,
                                               images=images_batch)['logits']
                    score_batch = [tensor_ for tensor_ in score_batch]
                    score.extend(score_batch)

            total_batch_num = bs * num_return_sequences * num_return_sequences_span
            ppo_train(ppo_trainer, ppo_query, ppo_response, score, ppo_images,
                      total_batch_num, bs)
            if index % 50 == 0:
                save_dir = os.path.join(
                    script_args.output_dir,
                    'epoch' + str(epoch) + '_' + str(index))
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                ppo_trainer._save_pretrained(save_dir)

        save_dir = os.path.join(script_args.output_dir,
                                'epoch' + str(epoch) + '_final')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        ppo_trainer._save_pretrained(save_dir)




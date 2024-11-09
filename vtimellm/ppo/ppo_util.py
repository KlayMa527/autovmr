import re
import torch
from vtimellm.mm_utils import tokenizer_image_token
from vtimellm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from vtimellm import conversation as conversation_lib
from vtimellm.model.vtimellm_llama_reward import VTimeLLMLlamaForSequenceClassification
from vtimellm.model.vtimellm_llama import VTimeLLMLlamaForCausalLM
from peft import PeftModel
import os


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def process_string(s):
    # 如果字符串是空格或者标点开头，将空格和标点去掉
    s = re.sub(r'^\W+|\W+$', '', s)

    # 如果字符串是小写开头，将字符串的开头变为大写
    if s and s[0].islower():
        s = s[0].upper() + s[1:]

    return s

def extract_query(sentence):
    'extrace description of video.'
    match = re.search(r'(.+?), from \d{2} to \d{2}\.', sentence)
    if match:
        return match.group(1)
    else:
        return None
def extract_time_span(sentence):
    match = re.search(r'(from \d{2} to \d{2})', sentence)
    if match:
        matched_string =  matched_string = match.group(1)
        span_match = re.search(r'from (\d{2}) to (\d{2})', matched_string)
        if span_match :

            start = int(span_match.group(1))
            end = int (span_match.group(2))
            return start, end, matched_string
        else:

            return None, None, matched_string
        
    else:
        return None, None, None
    

def generate_time_span_query(images, responses, tokenizer, num_return_sequences, prompt):
    span_image_list = []
    span_input_ids_list = []
    span_match_list = []
    span_prompt_list = []
    for index , response in enumerate(responses):
        match = extract_query(response)
        if match is None:
            continue
        new_prompt = prompt[index // num_return_sequences] + match  + ','
        new_input_ids = tokenizer_image_token(new_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        span_match_list.append(match)
        span_prompt_list.append(new_prompt)
        span_input_ids_list.append(new_input_ids)
        span_image_list.append(images[index // num_return_sequences,::])
    if all(x is not None and x.shape == span_image_list[0].shape
            for x in span_image_list):
        span_image_list = torch.stack(span_image_list)
    else:
        span_image_list = span_image_list
    return span_image_list, span_input_ids_list, span_prompt_list, span_match_list


def generate_reward_model_input_ids(reward_model_prompt, query, tokenizer):
    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], reward_model_prompt)
    conv.append_message(conv.roles[1], query)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
    return input_ids, prompt

def generate_reward_model_and_forward_input(images, responses, tokenizer, num_return_sequences, prompt, matches, reward_model_prompt):
    ppo_image_list = []
    ppo_input_ids_list = []
    ppo_match_list = []
    ppo_prompt_list = []
    ppo_query_list = []
    ppo_response_list = []

    reward_model_input_ids_list = []
    reward_model_images_list = []
    reward_model_prompt_list = []
    for index , response in enumerate(responses):
        start, end , match_span = extract_time_span(response)
        if start is None or end is  None:
            continue

        # generate reward model input
        reward_model_query = process_string(matches[index // num_return_sequences]) + '.'
        reward_model_input_ids, reward_model_query = generate_reward_model_input_ids(reward_model_prompt, reward_model_query, tokenizer)

        reward_model_input_ids_list.append(reward_model_input_ids)
        reward_model_images_list.append(images[index // num_return_sequences,::][start:end, ::])
        reward_model_prompt_list.append(reward_model_query)

        # generate ppo input
        ppo_query = prompt[index // num_return_sequences]
        ppo_response = ' ' + match_span + '.</s>'
        ppo_prompt = ppo_query + ppo_response
        ppo_input_ids = tokenizer_image_token(ppo_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        ppo_match_list.append(match_span)
        ppo_prompt_list.append(ppo_prompt)
        ppo_input_ids_list.append(ppo_input_ids)
        ppo_image_list.append(images[index // num_return_sequences,::])
        ppo_query_list.append(tokenizer_image_token(ppo_query, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda())
        ppo_response_list.append(tokenizer_image_token(ppo_response, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')[1:].cuda())
        # ppo_response_list.append(torch.tensor(tokenizer(ppo_response).input_ids))
    


    if all(x is not None and x.shape == ppo_image_list[0].shape
            for x in ppo_image_list):
        ppo_image_list = torch.stack(ppo_image_list)
    else:
        ppo_image_list = ppo_image_list

    if all(x is not None and x.shape == reward_model_images_list[0].shape
            for x in reward_model_images_list):
        reward_model_images_list = torch.stack(reward_model_images_list)
    else:
        reward_model_images_list = reward_model_images_list

    # padding reward model input
    reward_model_input_ids_list = torch.nn.utils.rnn.pad_sequence(
        reward_model_input_ids_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    reward_model_input_ids_list =  reward_model_input_ids_list[:, :tokenizer.model_max_length]

    return ppo_image_list, ppo_query_list,ppo_response_list, ppo_input_ids_list, ppo_prompt_list, ppo_match_list, reward_model_images_list, reward_model_input_ids_list, reward_model_prompt_list

def load_lora(model, lora_path):
    non_lora_trainables_path = os.path.join(lora_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_trainables_path):
        non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, lora_path)
    return model


def load_reward_model(args, stage2, stage3, reward_model):
    model = VTimeLLMLlamaForSequenceClassification.from_pretrained(
        args.model_base,  num_labels=1
    )
    model.to(torch.bfloat16)
    print('Loading reward model ... ...')
    print('Loading stage2 weights...')
    model = load_lora(model, stage2)
    print('Merging stage2 weights...')
    model = model.merge_and_unload()

    # load stage_3 lora
    print('Loading stage3 weights...')
    model = load_lora(model, stage3)
    print('Merging stage3 weights...')
    model = model.merge_and_unload()

    # load reward model
    print('Loading reward model...')
    model = load_lora(model, reward_model)
    print('Merging reward model weights...')
    model = model.merge_and_unload()
    print('Loading reward model finished.')

    model.get_model().initialize_vision_modules(args)
    model.get_model().mm_projector.to(torch.float16)

    return model



def load_pretrained_model(args, stage2=None, stage3=None):
    kwargs = {'torch_dtype': torch.float16}

    # model_path = os.path.expanduser(args.model_path)
    model_base = args.model_base


    # lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
    print('Loading VTimeLLM from base model...')
    model = VTimeLLMLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True)
    token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    if model.lm_head.weight.shape[0] != token_num:
        model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
        model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

    model.to(torch.bfloat16)


    if stage2 is not None:
        print('Loading stage2 weights...')
        model = load_lora(model, stage2)
        print('Merging stage2 weights...')
        model = model.merge_and_unload()
        if stage3 is not None:
            print('Loading stage3 weights...')
            model = load_lora(model, stage3)
            print('Merging stage3 weights...')
            model = model.merge_and_unload()
    
    
    # if hasattr(model.config, "max_sequence_length"):
    #     context_len = model.config.max_sequence_length
    # else:
    #     context_len = 2048

    return  model




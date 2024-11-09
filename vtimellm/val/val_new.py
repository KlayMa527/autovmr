import os
import json
import sys
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(root_dir)

import  argparse
from vtimellm.utils import disable_torch_init
from vtimellm.model.builder import load_pretrained_model
from vtimellm.mm_utils import VideoExtractor, tokenizer_image_token, KeywordsStoppingCriteria
from vtimellm.conversation import conv_templates, SeparatorStyle
from vtimellm.constants import IMAGE_TOKEN_INDEX
import torch
import re
import numpy as np
from tqdm import tqdm


def process_string(s):
    # 如果字符串是空格或者标点开头，将空格和标点去掉
    s = re.sub(r'^\W+|\W+$', '', s)
    if s and s[0].islower():
        s = s[0].upper() + s[1:]

    return s

def inference(model, image, query_1, query_2, tokenizer):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query_1)
    conv.append_message(conv.roles[1], query_2)
    prompt = conv.get_prompt()
    prompt = prompt[:-4] if prompt.endswith('</s>') else prompt
    input_ids = tokenizer_image_token(prompt,
                                      tokenizer,
                                      IMAGE_TOKEN_INDEX,
                                      return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer,
                                                 input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image[None,].cuda(),
            do_sample=True,
            temperature=0.05,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1295

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids'
        )
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:],
                                     skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs

def extract_from_to(text):
    # 正则表达式模式，寻找 "from xxx to xxx" 结构
    pattern = r'\b(\d{2})\b to \b(\d{2})\b'

    # 查找所有匹配的部分
    matches = re.findall(pattern, text)
    matches = [[int(item[0]), int(item[1])] for item in matches]
    # matches 是一个元组列表，每个元组包含两个分组：a 和 b
    return matches

def save_json(save_path, save_name, result):
    save_path = os.path.join(save_path, save_name)
    json_file = open(save_path, 'w')
    json.dump(result, json_file, indent=2)

def calculate_tiou(interval_pred, interval_true):
    start_pred, end_pred = interval_pred
    start_true, end_true = interval_true

    # Find the intersection.
    intersection_start = max(start_pred, start_true)
    intersection_end = min(end_pred, end_true)

    # Calculate the intersection duration.
    intersection_duration = max(0, intersection_end - intersection_start)

    # Calculate the union duration.
    union_duration = max(end_pred, end_true) - min(start_pred, start_true)

    # Calculate the tiou.
    tiou = intersection_duration / union_duration if union_duration != 0 else 0

    return tiou

def parse_args():
    parser = argparse.ArgumentParser(description="Activity Net")
    parser.add_argument("--clip_path",type=str,default='./checkpoints/clip/ViT-L-14.pt')
    parser.add_argument("--model_base",type=str,default='./checkpoints/vicuna-7b-v1.5')
    parser.add_argument("--pretrain_mm_mlp_adapter",type=str,default="./checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2",type=str,default="./checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3",type=str,default="./checkpoints/sft_ckp/vtimellm-vicuna-v1-5-7b-stage3_5")
    parser.add_argument("--ppo_stage",type=str,default="./checkpoints/ppo_ckp/")
    parser.add_argument("--annotation_path",type=str,default='./dataset/val_data/charades_sta/charades_sta_val.json')
    parser.add_argument("--task",type=str,default='ac')
    parser.add_argument('--log_path',type=str,default='./output/val/')
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--ppo', action="store_true")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.device)
    disable_torch_init()

    # build dataset
    print('ppo:',args.ppo)


    if args.ppo:
        tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3, args.ppo_stage)
    else:
        tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3)

    model = model.cuda()
    model.to(torch.float16)

    if args.task == 'ac':
        feat_folder = './dataset/val_data/activity_net/feat/val/'
        annotation_path = './dataset/val_data/activity_net/activity_net_val.json'
        log_path = os.path.join(args.log_path,'activity_net')
    elif args.task == 'sta':
        feat_folder = './dataset/val_data/charades_sta/feat/'
        annotation_path = './dataset/val_data/charades_sta/charades_sta_val.json'
        log_path = os.path.join(args.log_path,'charades_sta')
    else:
        raise ValueError('error task type')

    data_path = annotation_path
    val_data = json.load(open(data_path))

    answer_list = []
    query_list = []
    tiou_list = []

    for i, val_item in tqdm(enumerate(val_data), total=len(val_data), desc='val time:'):
        duration = val_item['duration']
        id = val_item['id']
        start_gt = float(val_item['start_gt'])
        end_gt = float(val_item['end_gt'])
        query = process_string(val_item['query'])
        interval_true = [start_gt, end_gt]
        feat_path = os.path.join(feat_folder, f'{id}.npy')
        if os.path.isfile(feat_path):
            features = torch.from_numpy(np.load(feat_path)).cuda()
        else:
            print(feat_path)
            raise ValueError('miss path')
        prompt = "Can you provide a detailed introduction to the events that occurred in the video and provide corresponding time periods?"
        query = query + ', from'
        answer = inference(model, features, "<video>\n " + prompt, query, tokenizer)
        answer_list.append(answer)
        query_list.append(query)
        result = extract_from_to(answer)
        t_iou = ''
        if len(result) == 0:
            t_iou = -1
            tiou_list.append(-1)
        else:
            interval_pred = [item * (duration / 100) for item in result[0]]
            t_iou = calculate_tiou(interval_pred, interval_true)
            tiou_list.append(t_iou)

        print('num:', i)
        print('query:', query)
        print('answer:', answer)
        print('t_iou:', t_iou)

    threshold_list = [0.3, 0.5, 0.7]
    proportion_list = []
    # cal threshold
    for threshold in threshold_list:
        tiou_greater_list = [item for item in tiou_list if item > threshold]
        proportion = len(tiou_greater_list) / len(tiou_list)
        proportion_list.append(proportion)
    # make final list
    final_list = []
    for query, answer, tiou in zip(query_list, answer_list, tiou_list):
        all_item = query + ' '+ answer + ' tiou:' + str(tiou)
        final_list.append(all_item)    
    # save_data
    save_data = {
        "tiou_list": tiou_list,
        'answer_list': answer_list,
        'final_list': final_list,
        'porprotion': proportion_list,
        "tiou_mean": sum(tiou_list) / len(tiou_list),
    }

    if args.ppo:
        save_name = args.stage3.split('/')[-1] + '_' + args.ppo_stage.split('/')[-2] + '_' + args.ppo_stage.split('/')[-1] +'.json'
    else:
        save_name = args.stage3.split('/')[-1] + '.json'

    save_path = log_path
    save_json(save_path, save_name, save_data)

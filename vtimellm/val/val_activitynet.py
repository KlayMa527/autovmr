import os
import json
import sys
import torch
import clip
import argparse
import re
from tqdm import tqdm
import numpy as np

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
print(root_dir)
sys.path.append(root_dir)

from vtimellm.utils import disable_torch_init
from vtimellm.model.builder import load_pretrained_model
from vtimellm.mm_utils import VideoExtractor, tokenizer_image_token, KeywordsStoppingCriteria
from vtimellm.conversation import conv_templates, SeparatorStyle
from vtimellm.constants import IMAGE_TOKEN_INDEX
from moviepy.editor import VideoFileClip

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

tqdm.pandas()

def process_string(s):
    # 如果字符串是空格或者标点开头，将空格和标点去掉
    s = re.sub(r'^\W+|\W+$', '', s)
    if s and s[0].islower():
        s = s[0].upper() + s[1:]

    return s


def parse_args():
    parser = argparse.ArgumentParser(description="Activity Net")
    parser.add_argument("--clip_path",
                        type=str,
                        default='checkpoints/clip/ViT-L-14.pt')
    parser.add_argument("--model_base",
                        type=str,
                        default='checkpoints/vicuna-7b-v1.5')
    parser.add_argument(
        "--pretrain_mm_mlp_adapter",
        type=str,
        default="checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2",
                        type=str,
                        default="checkpoints/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3",
                        type=str,
                        default="checkpoints/vtimellm-vicuna-v1-5-7b-stage3_5")
    parser.add_argument("--feat_folder", type=str, default='/home/luoshu/VTimeLLM/dataset/activitynet/feat/val/')

    parser.add_argument("--ppo_stage",
                        type=str,
                        default="/home/luoshu/VTimeLLM/output/ppo/ppo_fromv5_exp_2/epoch0_final")
    parser.add_argument("--data_path",
                        type=str,
                        default="/home/luoshu/VTimeLLM/data_pre/activity_acption/dataset/val_2.json")
    parser.add_argument(
        "--annotation_path",
        type=str,
        default=
        '/home/luoshu/VTimeLLM/data_pre/activity_acption/dataset/val_2.json'
    )
    parser.add_argument('--log_path',
                        type=str,
                        default='/home/luoshu/VTimeLLM/output/val/from_v5/')
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--ppo', action="store_true")
    args = parser.parse_args()
    return args


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


def get_clip_feature(model, video_path):
    video_loader = VideoExtractor(N=100)
    _, images = video_loader.extract({'id': None, 'video': video_path})
    #define transform
    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])

    # print(images.shape) # <N, 3, H, W>
    images = transform(images / 255.0)
    images = images.to(torch.float16)
    with torch.no_grad():
        features = model.encode_image(images.to('cuda'))
    return features


def get_annotation(annotation_path):
    annotation_list = []
    annotation_data = open(annotation_path, 'r')
    for annotation_item in annotation_data.readlines():
        video_info, query = annotation_item.strip().split('##')
        query = process_string(query)
        video_name, start_time, end_time = video_info.split(' ')
        annotation_list.append({
            'video_name': video_name,
            'query': query,
            'start_time': start_time,
            'end_time': end_time
        })
    return annotation_list


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


def extract_from_to(text):
    # 正则表达式模式，寻找 "from xxx to xxx" 结构
    pattern = r'\b(\d{2})\b to \b(\d{2})\b'

    # 查找所有匹配的部分
    matches = re.findall(pattern, text)
    matches = [[int(item[0]), int(item[1])] for item in matches]
    # matches 是一个元组列表，每个元组包含两个分组：a 和 b
    return matches


def get_video_duration(video_path):
    with VideoFileClip(video_path) as video:
        return video.duration


def save_json(save_path, save_name, result):
    save_path = os.path.join(save_path, save_name)
    json_file = open(save_path, 'w')
    json.dump(result, json_file, indent=2)


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.device)
    # initialize model
    disable_torch_init()

    print(args.ppo)
    print(args.feat_folder)
    if args.ppo:
        tokenizer, model, context_len = load_pretrained_model(
            args, args.stage2, args.stage3, args.ppo_stage)
    else:
        tokenizer, model, context_len = load_pretrained_model(
            args, args.stage2, args.stage3)
    model = model.cuda()
    model.to(torch.float16)

    clip_model, _ = clip.load(args.clip_path)
    clip_model.eval()
    clip_model = clip_model.cuda()
    data_path = args.data_path
    feat_folder = args.feat_folder
    tiou_list = []
    answer_list = []
    query_list = []
    # load val data
    val_data = json.load(open(data_path))

    for id, data in tqdm(val_data.items()):
        feat_path = os.path.join(args.feat_folder, f"{id}.npy")
        duration = data['duration']
        if os.path.isfile(feat_path):
            features = torch.from_numpy(np.load(feat_path)).cuda()
        else:
            print('no exit file')
        for sentence_id, (timestamps, sentence) in enumerate(
                zip(data['timestamps'], data['sentences'])):
            sentence = process_string(sentence.strip().lower())
            gt_s = float(timestamps[0])
            gt_e = float(timestamps[1])
            interval_true = [gt_s, gt_e]
            prompt = "Can you provide a detailed introduction to the events that occurred in the video and provide corresponding time periods?"
            query = sentence + ', from'
            answer = inference(model, features, "<video>\n " + prompt, query,
                               tokenizer)
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

            print('query:', query)
            print('answer:', answer)
            print('t_iou:', t_iou)

    print(tiou_list)
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
    # define save name
    if args.ppo:
        save_name = args.stage3.split('/')[-1] + '_' + args.ppo_stage.split('/')[-2] + '_' + args.ppo_stage.split('/')[-1] +'.json'

    else:
        save_name = args.stage3.split('/')[-1] + '.json'

    save_path = args.log_path
    save_json(save_path, save_name, save_data)

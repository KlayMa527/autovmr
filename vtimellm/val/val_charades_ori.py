import os
import json
import sys
import torch
import clip
import argparse
import re
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..","..")
print(root_dir)
sys.path.append(root_dir)

from vtimellm.utils import disable_torch_init
from vtimellm.model.builder import load_pretrained_model
from vtimellm.mm_utils import VideoExtractor,tokenizer_image_token,KeywordsStoppingCriteria
from vtimellm.conversation import conv_templates, SeparatorStyle
from vtimellm.constants import IMAGE_TOKEN_INDEX
from moviepy.editor import VideoFileClip

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose,Resize,CenterCrop,Normalize

def parse_args():
    parser = argparse.ArgumentParser(description="Charades_STA")
    parser.add_argument("--clip_path", type=str, default='checkpoints/clip/ViT-L-14.pt')
    parser.add_argument("--model_base", type=str,default='checkpoints/vicuna-7b-v1.5')
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage3_2")
    parser.add_argument("--data_path", type=str, default="dataset/charades_sta/Charades_v1_480")
    parser.add_argument("--annotation_path", type=str, default='/home/luoshu01/VTimeLLM/dataset/charades_sta/Charades/charades_sta_test.txt' )
    parser.add_argument('--log_path',type=str,default='/home/luoshu01/VTimeLLM/vtimellm/val')
    parser.add_argument('--device',type=int ,default=0)
    args = parser.parse_args()
    return args

def inference(model, image, query_1,query_2,tokenizer):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query_1)
    conv.append_message(conv.roles[1], query_2)
    prompt = conv.get_prompt()
    prompt = prompt[:-4] if prompt.endswith('</s>') else prompt
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

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
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs




def get_clip_feature(model,video_path):
    video_loader = VideoExtractor(N=100)
    _, images = video_loader.extract({'id': None, 'video': video_path})
    #define transform
    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # print(images.shape) # <N, 3, H, W>
    images = transform(images / 255.0)
    images = images.to(torch.float16)
    with torch.no_grad():
        features = model.encode_image(images.to('cuda'))
    return features
    
def get_annotation(annotation_path):
    annotation_list = []
    annotation_data = open(annotation_path,'r')
    for annotation_item in annotation_data.readlines():
        video_info, query = annotation_item.strip().split('##')
        query = query[:-1] if query.endswith('.') else query
        video_name, start_time, end_time = video_info.split(' ')
        annotation_list.append({
            'video_name': video_name,
            'query':query,
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

    # Calculate the tIoU.
    tIoU = intersection_duration / union_duration if union_duration != 0 else 0

    return tIoU

def extract_from_to(text):
    # 正则表达式模式，寻找 "from xxx to xxx" 结构
    pattern = r'(\d+) to (\d+)'

    # 查找所有匹配的部分
    matches = re.findall(pattern, text)
    matches = [[int(item[0]), int(item[1])] for item in matches]
    # matches 是一个元组列表，每个元组包含两个分组：a 和 b
    return matches

def get_video_duration(video_path):
    with VideoFileClip(video_path) as video:
        return video.duration

def save_json(save_path,save_name,result):
    save_path = os.path.join(save_path,save_name)
    json_file = open(save_path,'w')
    json.dump(result,json_file,indent=2)

if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.device)
    # # initialize model 
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3)
    model = model.cuda()
    model.to(torch.float16)

    clip_model, _ = clip.load(args.clip_path)
    clip_model.eval()
    clip_model = clip_model.cuda()
    data_path = args.data_path
    annotation_path = args.annotation_path
    annotation_list = get_annotation(annotation_path)
    tIoU_list = []
    answer_list = []
    for i, item in enumerate(annotation_list):
        video_path = os.path.join(args.data_path,item['video_name']+'.mp4')
        gt_s = float(item['start_time'])
        gt_e = float(item['end_time'])
        video_duration = get_video_duration(video_path)
        interval_true = [gt_s, gt_e]
        prompt = "when the "+ item['query'] +" in the video?"
        query = None
        features = get_clip_feature(clip_model, video_path)
        answer = inference(model, features, "<video>\n " + prompt, query, tokenizer)
        answer_list.append(answer)
        result = extract_from_to(answer)
        if len(result) == 0:
            tIoU_list.append(-1)
        else:
            interval_pred = [item * (video_duration / 100) for item in result[0]] 
            tIoU_list.append(calculate_tiou(interval_pred,interval_true))
        print('num:',i)
        print('query:',query)
        print('answer:',answer)

    print(tIoU_list)
    # define threshold
    threshold_list = [0.3, 0.5, 0.7]
    proportion_list = []
    # cal threshold
    for threshold in threshold_list:
        tIou_greater_list = [item for item in tIoU_list if item > threshold]
        proportion = len(tIou_greater_list) / len(tIoU_list)
        proportion_list.append(proportion)
    # save_data
    save_data = {
        "tIoU_list": tIoU_list,
        'porprotion': proportion_list,
        'answer_list': answer_list
    }
    # define save name
    save_name = args.stage3.split('/')[-1] + '.json'
    save_path = args.log_path
    save_json(save_path, save_name, save_data)







    





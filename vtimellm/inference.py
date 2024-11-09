import os
import sys
import argparse
import torch

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
print(root_dir)
sys.path.append(root_dir)


from vtimellm.constants import IMAGE_TOKEN_INDEX
from vtimellm.conversation import conv_templates, SeparatorStyle
from vtimellm.model.builder import load_pretrained_model, load_lora
from vtimellm.utils import disable_torch_init
from vtimellm.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, VideoExtractor
from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
from easydict import EasyDict as edict
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import numpy as np
import clip


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



def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--clip_path", type=str, default="checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--model_base", type=str, default="/path/to/vicuna-7b-v1.5")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage3")
    parser.add_argument("--video_path", type=str, default="images/baby.mp4")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    args.clip_path = '/home/luoshu/VTimeLLM/checkpoints/clip/ViT-L-14.pt'
    args.model_base = '/home/luoshu/VTimeLLM/checkpoints/vicuna-7b-v1.5'
    args.pretrain_mm_mlp_adapter = '/home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin'
    args.stage2 = '/home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage2'
    args.stage3 = '/home/luoshu/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3'
    args.video_path = '/home/luoshu/VTimeLLM/dataset/charades_sta/Charades_v1_480/3MSZA.mp4'

    torch.cuda.set_device(3)
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3)
    model = model.cuda()
    # model.get_model().mm_projector.to(torch.float16)
    model.to(torch.float16)

    clip_model, _ = clip.load(args.clip_path)
    clip_model.eval()
    clip_model = clip_model.cuda()

    video_loader = VideoExtractor(N=100)
    _, images = video_loader.extract({'id': None, 'video': args.video_path})

    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # print(images.shape) # <N, 3, H, W>
    images = transform(images / 255.0)
    images = images.to(torch.float16)
    with torch.no_grad():
        features = clip_model.encode_image(images.to('cuda'))

    # query = "describe the video."
    prompt = "Can you pinpoint the moment when the person flipped the light switch near the door in the video?"
    query = None
    print("prompt: ", prompt)
    print("answer: ", inference(model, features, "<video>\n " + prompt, query,tokenizer))



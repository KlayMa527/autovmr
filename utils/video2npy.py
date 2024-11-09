import os
import json
import shutil
import numpy as np
import clip
import torch
import decord
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from tqdm import tqdm

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC


class VideoExtractor():
    """Dataset for supervised fine-tuning."""

    def __init__(self, N=100):
        self.N = N

    def extract(self, data):
        video_path = data['video']
        id = data['id']

        try:
            video_reader = decord.VideoReader(video_path)
            total_frames = len(video_reader)
            start = 0
            end = total_frames - 1

            split = data.get('split', None)
            if split is not None:
                fps = video_reader.get_avg_fps()
                start = max(int(fps * split[0]), 0)
                end = min(int(fps * split[1]), total_frames - 1)
            sampled_indices = np.linspace(start, end, self.N, dtype=np.int32)
            sampled_frames = video_reader.get_batch(sampled_indices).asnumpy()
        except Exception as e:
            print(e)
            return None, torch.zeros(1)

        images = torch.from_numpy(sampled_frames.transpose((0, 3, 1, 2)))
        return id, images

torch.cuda.set_device(2)
video_dir = '/home/luoshu/VTimeLLM/video'
save_path = '/home/luoshu/VTimeLLM/video'
video_name = os.listdir(video_dir)
clip_model, _ = clip.load('/home/luoshu/VTimeLLM/checkpoints/clip/ViT-L-14.pt')
clip_model.eval()
clip_model = clip_model.cuda()
video_loader = VideoExtractor(N=100)
transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

for index, video in tqdm(enumerate(video_name), total=len(video_name)):
    video_path = os.path.join(video_dir, video)
    print(video_path)
    if os.path.isfile(video_path):
        _, images = video_loader.extract({'id': None, 'video': video_path})
        images = transform(images / 255.0)
        with torch.no_grad():
            features = clip_model.encode_image(images.to('cuda')).cpu().numpy()
        npy_file_path = os.path.join(save_path, f"{video.split('.')[0]}.npy")
        np.save(npy_file_path, features)
        print(index)
print('done')
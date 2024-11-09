import json
import os


stage3_data_path = '/home/luoshu01/VTimeLLM/feat/stage3_clip_feat/'
stage3_data = json.load(open('/home/luoshu01/VTimeLLM/data/stage3.json','r'))

stage3_data_list = []

for data in os.listdir(stage3_data_path):
    stage3_data_list.append(data.split('.')[0])

miss_count = 0
total_count = 0
counted = []

for data in stage3_data:
    if data['source'] == 'anet' and 'meta' in data.keys():
        id = data['id']
        if id not in counted:
            total_count += 1
            counted.append(id)
            if id not in stage3_data_list:
                miss_count += 1
print(miss_count)
print(total_count)
import json

data_path = '/home/luoshu01/VTimeLLM/data/stage3.json'

data = json.load(open(data_path,'r'))
for item in data:
    if '<image>' in item['conversations'][0]['value']:
        print(item)
        break
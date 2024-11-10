import json

result_path = './output/val/vtimellm-vicuna-v1-5-7b-stage3_1.json'

result_data = json.load(open(result_path))

tiou_list = result_data['tIoU_list']

tag_list = [0] * 10

for item in tiou_list:
    tag_list[int(item * 100 // 10)] += 1

print(len(tiou_list))
print(tag_list)


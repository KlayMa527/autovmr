import re
import json
# def extract_from_to(text):
#     # 正则表达式模式，寻找 "from xxx to xxx" 结构
#     pattern = r' (\w+) to (\w+)'

#     # 查找所有匹配的部分
#     matches = re.findall(pattern, text)

#     # matches 是一个元组列表，每个元组包含两个分组：a 和 b
#     return matches


# data = json.load(open('/home/luoshu01/VTimeLLM/vtimellm/val/vtimellm-vicuna-v1-5-7b-stage3.json','r'))

# tIoU_list = data['tIoU_list']

# threshold_list = [0.3, 0.5, 0.7]
# proportion_list = []
# # cal threshold
# for threshold in threshold_list:
#     tIou_greater_list = [item for item in tIoU_list if item > threshold]
#     proportion = len(tIou_greater_list) / len(tIoU_list)
#     proportion_list.append(proportion)

# print(proportion_list)


# average = sum(tIoU_list) / len(tIoU_list)

# # Print the average
# print("Average:", average)

# data = json.load(open('/home/luoshu01/VTimeLLM/data_pre/new_stage3_v1.json'))
# print(len(data))

# lt = range(0,10)
# samples = random.sample(lt,5)
# print(samples)
# instances = [
#     {"input_ids": [101, 102, 103, 104], "labels": [1, 0, 1, 0]},
#     {"input_ids": [201, 202, 203, 204], "labels": [0, 1, 0, 1]},
#     {"input_ids": [301, 302, 303, 304], "labels": [1, 1, 0, 0]}
# ]

# input_ids, labels = tuple([instance[key] for instance in instances]
#                                 for key in ("input_ids", "labels"))
# input_ids_2 = tuple(instance['input_ids'] for instance in instances)
# print(input_ids_2)

import json
data = json.load(open('/home/luoshu/VTimeLLM/data_pre/reward_model_train_v2.json'))

len_ = len(data)
print(len(data))
print(len_ / 256)

'''
Given a set of pre-trained video features and their corresponding textual descriptions, your task is to develop a reward model that accurately assesses the alignment between each video and its description. The input to your model will be a feature vector representing the video and a text string containing its description. Your model should output a score that reflects the degree of match between the video content and the text.

1. Start by pre-processing the textual descriptions to ensure they are clean and normalized, suitable for model input.
2. Use an attention mechanism to focus on key elements within the video features and text descriptions that are most relevant for assessing their match.
3. Employ a neural network architecture capable of handling multimodal inputs (e.g., CNN for video features and Transformer for text) to learn complex relationships between the two modalities.
4. Fine-tune your model using a dataset of video-text pairs with annotated match scores to learn the nuances of what constitutes a good match.
5. Implement a scoring mechanism that quantifies the degree of match, considering both the semantic content and context of the video and text.
6. Regularly evaluate the model on a validation set to monitor its performance and prevent overfitting. Use feedback from these evaluations to iteratively improve the model.
7. Your ultimate goal is to create a reward model that is robust, interpretable, and can generalize well to unseen video-text pairs, maximizing the accuracy of the match assessment.

'''

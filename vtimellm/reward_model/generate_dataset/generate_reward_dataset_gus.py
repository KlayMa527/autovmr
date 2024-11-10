import numpy as np
import re
import json
import os
import random


def calculate_tiou(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return intersection / union if union != 0 else 0


def remove_punctuation_and_spaces(s):
    """
    去除字符串开头和结尾的标点符号和空格。
    """
    return re.sub(r'^\W+|\W+$', '', s)


def sample_within_tiou_range(duration, s1, e1, tiou_range):
    original_action = (s1, e1)
    try_times = 10000000
    cur_time = 0
    while True:
        # # 如果片段太短，跳过采样
        # if (e1 - s1) < 1:
        #     return None

        # 随机采样中心点和持续时间
        center = np.random.uniform(0, duration)
        sampled_duration = np.random.uniform(0, duration)

        start = round(max(0, min(duration, center - sampled_duration / 2)), 2)
        end = round(max(0, min(duration, center + sampled_duration / 2)), 2)
        sampled_action = (start, end)

        # 计算tIoU
        tiou = calculate_tiou(original_action, sampled_action)

        # 检查tIoU是否在指定范围内
        if tiou_range[0] < tiou <= tiou_range[1]:
            return sampled_action, tiou
        # 判断尝试次数
        if cur_time > try_times:
            return None, None
        cur_time += 1


def sample_actions(duration, s1, e1, tiou_ranges):
    samples = []
    t_IoU_lt = []
    for tiou_range in tiou_ranges:
        sample, t_IoU = sample_within_tiou_range(duration, s1, e1, tiou_range)
        if sample is not None:
            samples.append(sample)
            t_IoU_lt.append(t_IoU)
        else:
            print(duration, s1, e1)
    return samples,t_IoU_lt


def sample_val_dataset(lst):
    train_data = []
    val_data = []

    data_length = len(lst)
    total_index = range(0, data_length)
    sample_index = random.sample(total_index, data_length // 10)
    for i in range(data_length):
        if i not in sample_index:
            train_data.append(lst[i])
        else:
            val_data.append(lst[i])
    return train_data, val_data

def sample_gaussian(duration, s_gt, e_gt, sample_nums=2, scale=10):
    sample_list = []
    tiou_list = []
    while True:
        s_sample = np.random.normal(loc=s_gt, scale=scale)
        e_sample = np.random.normal(loc=e_gt, scale=scale)

        s_sample = round(max(0, min(duration, s_sample)), 2)
        e_sample = round(max(0, min(duration, e_sample)), 2)
        if s_sample < e_sample:
            sample_list.append([s_sample,e_sample])
            tiou_list.append(calculate_tiou([s_gt, e_gt],[s_sample, e_sample]))
        if len(sample_list) == sample_nums:
            break
    return sample_list, tiou_list





def cal_se_length(e_gt, s_gt):
    count = [0, 0, 0, 0]
    if (e_gt - s_gt) < 1:
        count[0] += 1
    elif (e_gt - s_gt) >= 1 and (e_gt - s_gt) < 5:
        count[1] += 1
    elif (e_gt - s_gt) >= 5 and (e_gt - s_gt) < 10:
        count[2] += 1
    else:
        count[3] += 1
    return None


json_path = './data_pre/new_stage3_v5.json'
save_train_path = './data_pre/reward_model_data/from_v5/reward_model_train_v_gus.json'
save_val_path = './data_pre/reward_model_data/from_v5/reward_model_val_v_gus.json'
save_miss_id_path = './data_pre/reward_model_data/from_v5/miss_id_v_gus.json'

json_data = json.load(open(json_path, 'r'))

save_data = []
miss_id = []
tiou_list_all = []

name_list = os.listdir('./feat/stage3_clip_feat/')
for index, item in enumerate(json_data):
    id = item['id']
    if (id + '.npy') not in name_list:
        continue
    source = item['source']
    query = remove_punctuation_and_spaces(
        item['conversations'][1]['value'].split(', from <s0>')[0])
    s_gt = item['meta']['token']['<s0>']
    e_gt = item['meta']['token']['<e0>']
    duration = item['meta']['duration']

    human_query = "<video>\n" + 'Can you provide a detailed introduction to the events that occurred in the video?'
    gpt_query = query + '.'
    human_dict = {'from': 'human', 'value': human_query}
    gpt_dict = {'from': 'gpt', 'value': gpt_query}
    conversation = [human_dict, gpt_dict]
    sample_times = 1
    sample_nums = 3
    scale = 5
    samples, tiou_lt = sample_gaussian(duration,s_gt,e_gt,sample_nums=sample_nums,scale=scale)
    tiou_list_all += tiou_lt
    # samples, tiou_lt = sample_actions(duration, s_gt, e_gt)
    for _ in range(sample_times):
        for i in range(sample_nums):
            for j in range(i+1, sample_nums):
                if tiou_lt[i] > tiou_lt[j]:
                    chosen = samples[i]
                    rejected = samples[j]
                    chosen_tiou = tiou_lt[i]
                    rejected_tiou = tiou_lt[j]
                else:
                    chosen = samples[j]
                    rejected = samples[i]
                    chosen_tiou = tiou_lt[j]
                    rejected_tiou = tiou_lt[i]
                save_item = dict(id=id,
                                query=query,
                                duration=duration,
                                s_gt=s_gt,
                                e_gt=e_gt,
                                chosen=chosen,
                                rejected=rejected,
                                chosen_tiou=chosen_tiou,
                                rejected_tiou=rejected_tiou,
                                conversations=conversation)
                save_data.append(save_item)
    print(index)
train_data, val_data = sample_val_dataset(save_data)

print(len(train_data))
print(len(val_data))

# train_file = open(save_train_path, 'w')
# val_file = open(save_val_path, 'w')

# json.dump(train_data, train_file, indent=2)
# json.dump(val_data, val_file, indent=2)

tag = [0] * 11
for item in tiou_list_all:
    tag[int(item * 100 // 10)] += 1
print(tag)
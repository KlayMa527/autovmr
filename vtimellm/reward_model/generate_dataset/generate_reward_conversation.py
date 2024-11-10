import json
import os

json_data = json.load(open('./data_pre/reward_model_val_v2.json'))
save_path = './data_pre/reward_model_val_v2_c_v2.json'
save_data = []

for data in json_data:
    # human_query = "<video>\n" + 'Do you think the video matches the query ? The query is ' + data['query'] + '.'
    # gpt_query = ' '
    human_query  = "<video>\n" + 'Can you provide a detailed introduction to the events that occurred in the video?'
    gpt_query = data['query'] + '.'
    human_dict = {
        'from' : 'human',
        'value' : human_query
    }
    gpt_dict = {
        'from' : 'gpt',
        'value' : gpt_query
    }
    conversation = [human_dict,gpt_dict]
    data['conversations'] = conversation
    save_data.append(data)

save_file = open(save_path, 'w')
json.dump(save_data,save_file,indent=2)
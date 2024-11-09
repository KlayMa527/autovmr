import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
IGNORE_INDEX = -100

def tokenizer_image_token(text, tokenizer):
    # 自定义函数，用于处理图片相关的tokenization
    pass

conv = {
    "sep": "  ",
    "sep2": "</s>",
    "roles": ["USER", "ASSISTANT"]
}
sep = conv["sep"] + conv["roles"][1] + ": "

conversations = ["A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <video>\nCould you give a quick summary of what's happening in the video? ASSISTANT: Of course! The video depicts a man preparing to run down a track. He exhibits remarkable speed as he sprints down the path. Finally, he takes a powerful leap, landing gracefully in the sand.</s>USER: I'm interested in the man's speed during the event from 14 to 54. Could you provide more information? ASSISTANT: Certainly! The man demonstrates exceptional speed during the event from 14 to 54, showcasing his agility and athleticism as he rapidly covers the distance on the track.</s>USER: Can you tell me when the man lands in the sand after his jump from 54 to 99? ASSISTANT: Yes, the man lands in the sand during the interval from 54 to 99, highlighting his impressive technique and coordination.</s>USER: What might be the purpose of the man's athletic display? ASSISTANT: The man's athletic display can be seen as an impressive feat of physical prowess, showcasing his skill and agility to the viewer.</s>"]

has_iamge = True
# 初始化目标数据列表
targets = [torch.zeros(tokenizer.model_max_length, dtype=torch.long) for _ in range(len(conversations))]

for conversation, target in zip(conversations, targets):
    total_len = int(target.ne(tokenizer.pad_token_id).sum())

    rounds = conversation.split(conv["sep2"])
    cur_len = 1
    target[:cur_len] = IGNORE_INDEX
    for i, rou in enumerate(rounds):
        if rou == "":
            break

        parts = rou.split(sep)
        if len(parts) != 2:
            break
        parts[0] += sep

        if has_image:
            round_len = len(tokenizer_image_token(rou, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
        else:
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

        target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

        cur_len += round_len
    target[cur_len:] = IGNORE_INDEX

    if cur_len < tokenizer.model_max_length:
        if cur_len != total_len:
            target[:] = IGNORE_INDEX
            print(
                f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                f" (ignored)"
            )
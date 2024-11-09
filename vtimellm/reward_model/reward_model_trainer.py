from typing import Any, Dict, Tuple, Union
import warnings
from torch._tensor import Tensor
from torch.nn.modules import Module
from transformers import PreTrainedModel
from trl import RewardTrainer
import torch
import torch.nn as nn
class VtimeLlmRewardTrainer(RewardTrainer):
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_reward_data_collator:
            warnings.warn(
                "The current compute_loss is implemented for RewardDataCollatorWithPadding,"
                " if you are using a custom data collator make sure you know what you are doing or"
                " implement your own compute_loss method."
            )

        rewards_chosen = model(
            input_ids = inputs['input_ids_chosen'],
            attention_mask = inputs['attention_mask_chosen'],
            labels = None,
            images =  inputs['images_chosen'],
            return_dict = True
        )['logits']
        rewards_rejected = model(
            input_ids = inputs['input_ids_rejected'],
            attention_mask = inputs['attention_mask_rejected'],
            labels = None,
            images =  inputs['images_rejected'],
            return_dict = True
        )['logits']
        # calculate loss, optionally modulate with margin
        if "margin" in inputs:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = torch.norm(param.grad)
        #         if grad_norm > 1:
        #             print(f"梯度爆炸检测在 {name}: 梯度大小 {grad_norm},after_loss")

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss


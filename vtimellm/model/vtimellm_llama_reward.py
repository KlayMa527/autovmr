import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import List, Optional, Tuple, Union
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM,  AutoModelForSequenceClassification,LlamaForSequenceClassification

from transformers.modeling_outputs import CausalLMOutputWithPast
from .vtimellm_arch_reward import VTimeLLMMetaModel, VTimeLLMMetaForCausalLM

class ScoreClassificationModel(nn.Module):
    def __init__(self):
        super(ScoreClassificationModel, self).__init__()
        # 第一个全连接层
        self.fc1 = nn.Linear(4096, 1024)
        # 批量归一化层
        self.bn1 = nn.BatchNorm1d(1024)
        # Dropout层
        self.dropout = nn.Dropout(0.5)
        # 第二个全连接层（二分类输出）
        self.fc2 = nn.Linear(1024, 1)

        # 应用权重初始化
        self._initialize_weights()

    def forward(self, x):
        # 假设x是Transformer的输出，形状为 [batch_size, 4096]
        x = F.relu(self.fc1(x))  # 应用ReLU激活函数
        x = self.dropout(x)      # 应用Dropout
        x = self.fc2(x)          # 最终二分类输出
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He初始化
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x 的形状: [batch_size, batch_max_len, input_dim]
        lstm_out, _ = self.lstm(x)
        # 选择LSTM输出的最后一个时间步
        last_time_step = lstm_out[:, -1, :]
        output = self.fc(last_time_step)
        return output


# define VTimeLLMConfig 
class VTimeLLMConfig(LlamaConfig):
    model_type = "VTimeLLM_classification"

# define VTimeLLMLlamaModel
class VTimeLLMLlamaModel(LlamaModel, VTimeLLMMetaModel):
    config_class = VTimeLLMConfig

    def __init__(self, config: LlamaConfig):
        super(VTimeLLMLlamaModel, self).__init__(config)

class VTimeLLMLlamaForSequenceClassification(LlamaForSequenceClassification, VTimeLLMMetaForCausalLM):
    config_class = VTimeLLMConfig

    def __init__(self, config):
        super(LlamaForSequenceClassification, self).__init__(config)
        self.model = VTimeLLMLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size

        # self.score = nn.Linear(config.hidden_size, 1, bias=True)

        # self.score =  nn.Sequential(
        #     nn.Linear(config.hidden_size, 256, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(256, 1, bias=False)
        # )
        
        # self.score = LSTMModel(input_dim=4096, hidden_dim=128, output_dim=1, num_layers=2)

        # Initialize weights and apply final processing
        self.score = ScoreClassificationModel()
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print(len(input_ids))
        # print(input_ids[0].shape)
        # print(images.shape)
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("VTimeLLM_classification", VTimeLLMConfig)
AutoModelForSequenceClassification.register(VTimeLLMConfig, VTimeLLMLlamaForSequenceClassification)

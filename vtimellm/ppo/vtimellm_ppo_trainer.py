import torch
import time
import numpy as np
from typing import List, Callable, Optional
import math
import torch.nn.functional as F
from trl import PPOTrainer, PreTrainedModelWrapper
from trl.core import PPODecorators,  convert_to_scalar, stack_dicts, stats_to_np, masked_whiten, WANDB_PADDING
from vtimellm.constants import IMAGE_TOKEN_INDEX, IMAGE_TOKEN_REPLACE_INDEX

# def process_image_token_index(logits, labels, image_token_index=IMAGE_TOKEN_INDEX, image_token_replace_index=IMAGE_TOKEN_REPLACE_INDEX):
#     labels = labels.clone()
#     masked_indices = (labels == image_token_index).nonzero(as_tuple=False)
#     new_logits_lt = []
#     for index in masked_indices:
#         row_index= index[0]
#         start_index = index[1]
#         end_index = start_index + 100
#         subtensor_before = logits[row_index, :start_index, :].unsqueeze(0)   # x 前的子张量
#         subtensor = logits[row_index, start_index:end_index, :]
#         averaged_subtensor = subtensor.mean(dim=0, keepdim=True).unsqueeze(0)
#         subtensor_after = logits[row_index, end_index:, :].unsqueeze(0)   # x+100 后的子张量
#         processed_subtensor = torch.cat((subtensor_before, averaged_subtensor, subtensor_after), dim=1)  # 拼接处理后的子张量
#         labels[row_index, start_index] = image_token_replace_index

#         new_logits_lt.append(processed_subtensor)
#     new_logits_lt = torch.cat(new_logits_lt, dim=0)

#     return new_logits_lt, labels

# def process_image_token_index(labels, image_token_index=IMAGE_TOKEN_INDEX, image_token_replace_index=IMAGE_TOKEN_REPLACE_INDEX):
#     labels = labels.clone()
#     indices = torch.nonzero(labels == image_token_index)
#     new_labels_list = []
#     if indices.numel() > 0:
#         for index in indices:
#             row, col = index.tolist()
#             pre = labels[row, :col].unsqueeze(0)
#             sub = torch.full((1,100), image_token_index).cuda()
#             after = labels[row, col + 1:].unsqueeze(0)
#             new_labels = torch.cat((pre, sub, after), dim=1)
#             new_labels_list.append(new_labels)
#     new_labels_list = torch.cat(new_labels_list, dim=0)

#     return new_labels_list

def process_image_token_index(labels, image_token_index=IMAGE_TOKEN_INDEX, replacement_length=100):
    labels = labels.clone()
    indices = (labels == image_token_index).nonzero(as_tuple=True)
    
    if not indices[0].size(0):
        return labels

    new_labels_list = []
    for row, col in zip(*indices):
        new_label = torch.cat([
            labels[row, :col],
            torch.full((replacement_length,), image_token_index, device=labels.device),
            labels[row, col + 1:]
        ])
        new_labels_list.append(new_label.unsqueeze(0))

    return torch.cat(new_labels_list, dim=0)


def logprobs_from_logits(logits: torch.Tensor,
                         labels: torch.Tensor,
                         gather: bool = True,
                         image_token_index=IMAGE_TOKEN_INDEX) -> torch.Tensor:
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)

    if not gather:
        return logp
        
    labels = process_image_token_index(labels=labels)
    # logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    
    # 替换labels中的无效索引值(-200)为0，保证索引有效
    valid_labels = torch.where(labels == image_token_index, torch.zeros_like(labels), labels)

    # 使用处理过的有效labels进行gather操作
    logpy = torch.gather(logp, 2, valid_labels.unsqueeze(2)).squeeze(-1)

    # 将原来是无效索引的位置结果置为一个特定的值，例如0或者负无穷（依据实际需求）
    logpy = torch.where(labels == image_token_index, torch.full_like(logpy, float(0.0)), logpy)

    return logpy


# def logprobs_from_logits(logits: torch.Tensor,
#                          labels: torch.Tensor,
#                          gather: bool = True,
#                          min_batch_size: int = 1) -> torch.Tensor:
#     """
#     Compute log probabilities from logits, handling large batch sizes by processing in smaller chunks.

#     """
#     if gather:
#         labels = process_image_token_index(labels=labels)

#     n_batches = max(logits.size(0) // min_batch_size,1)  # Calculate how many chunks are needed
#     batch_size = logits.size(0) // n_batches  # Compute the actual batch size to use

#     logpy_list = []  # List to store the chunks of log probabilities

#     for i in range(n_batches):
#         # Get the chunk of logits and labels
#         logits_chunk = logits[i * batch_size:(i + 1) * batch_size]

#         # Compute log softmax for the chunk
#         logp_chunk = F.log_softmax(logits_chunk, dim=2)

#         if gather:
#             labels_chunk = labels[i * batch_size:(i + 1) * batch_size]
#             # Gather the log probabilities for the correct labels
#             logpy_chunk = torch.gather(logp_chunk, 2,
#                                        labels_chunk.unsqueeze(2)).squeeze(-1)
#         else:
#             logpy_chunk = logp_chunk

#         logpy_list.append(logpy_chunk)

#     # Concatenate the chunks back into one tensor
#     logpy = torch.cat(logpy_list, dim=0)

#     return logpy


class VtimeLLMPPOTrainer(PPOTrainer):
    def _generate_batched(
        self,
        model: PreTrainedModelWrapper,
        query_tensors: List[torch.Tensor],
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        pad_to_multiple_of: int = None,
        remove_padding: bool = True,
        **generation_kwargs,
    ):
        outputs = []
        # define num_return_sequences
        if 'num_return_sequences' in generation_kwargs.keys():
            num_return_sequences = generation_kwargs['num_return_sequences']
        else:
            num_return_sequences = 1
        # define image batch
        if 'images' in generation_kwargs.keys():
            images_all = generation_kwargs['images']

        padding_side_default = self.tokenizer.padding_side
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        # in case we have fewer examples than bs
        batch_size = min(len(query_tensors), batch_size)

        for i in range(0, len(query_tensors), batch_size):
            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()

            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensors), i + batch_size)

            batch = query_tensors[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}
            if 'images' in generation_kwargs.keys():
                images = images_all[i:end_index,::]
                generation_kwargs['images'] = images

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(self.current_device)

            generations = self.accelerator.unwrap_model(model).generate(**padded_inputs, **generation_kwargs)

            for generation, mask in zip(generations, padded_inputs["attention_mask"].repeat_interleave(num_return_sequences,dim=0)):
                if not self.is_encoder_decoder:
                    output = generation[(1 - mask).sum() :]  # remove padding
                else:
                    output = generation

                if not return_prompt and not self.is_encoder_decoder:
                    output = output[(mask).sum() :]  # remove prompt

                if remove_padding and self.tokenizer.eos_token_id in output:
                    pad_mask = output == self.tokenizer.eos_token_id
                    pad_start = torch.nonzero(pad_mask, as_tuple=False)[0, 0].item()
                    output = output[: pad_start + 1]  # keep the eos token at the end

                outputs.append(output)

        self.tokenizer.padding_side = padding_side_default
        return outputs




    @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        response_masks: Optional[List[torch.LongTensor]] = None,
        images: List[torch.FloatTensor] = None,
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            response_masks (List[`torch.FloatTensor`], *optional*)):
                List of tensors containing masks of the response tokens.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        bs = self.config.batch_size

        queries, responses, scores, response_masks = self._step_safety_checker(
            bs, queries, responses, scores, response_masks
        )
        scores = torch.tensor(scores, device=self.current_device)
        if self.config.use_score_scaling:
            # Score scaling
            scores_mean, scores_std = self.running.update(scores)
            tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
            score_scaling_factor = self.running.std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
            if self.config.use_score_norm:
                scores = (scores - self.running.mean.to(**tensor_to_kwargs)) / score_scaling_factor
            else:
                scores /= score_scaling_factor

        if self.config.score_clip is not None:
            # Score clipping
            scores_dtype = scores.dtype
            scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip).to(dtype=scores_dtype)

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = scores.mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        timing = dict()
        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses, images)

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_attention_mask"],
                    dim=1,
                    pad_index=0,
                    pad_first=pad_first,
                )

        model_inputs_names = list(model_inputs.keys())

        full_kl_penalty = self.config.kl_penalty == "full"

        with torch.no_grad():
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                self.model,
                queries,
                responses,
                model_inputs,
                response_masks=response_masks,
                return_logits=full_kl_penalty,
            )
            with self.optional_peft_ctx():
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                    self.model if self.is_peft_model else self.ref_model,
                    queries,
                    responses,
                    model_inputs,
                    return_logits=full_kl_penalty,
                )

        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
                ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

                rewards, non_score_reward, kls = self.compute_rewards(
                    scores, active_full_logprobs, ref_full_logprobs, masks
                )
            else:
                rewards, non_score_reward, kls = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values, advantages, returns = self.compute_advantages(values, rewards, masks)
            timing["time/ppo/compute_advantages"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(model_inputs)

        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"],
                            mini_batch_dict["values"],
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"],
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                        )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
            kls=kls,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats


    def prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor, images: torch.Tensor):
        if self.is_encoder_decoder:
            input_data = self.data_collator(
                [{"input_ids": q, "attention_mask": torch.ones_like(q)} for q in queries]
            ).to(self.current_device)

            decoder_inputs = self.data_collator(
                [{"input_ids": r, "attention_mask": torch.ones_like(r)} for r in responses]
            ).to(self.current_device)

            input_data["decoder_input_ids"] = decoder_inputs["input_ids"]
            input_data["decoder_attention_mask"] = decoder_inputs["attention_mask"]
        else:
            input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
            input_data = self.data_collator(
                [{"input_ids": ids, "attention_mask": torch.ones_like(ids)} for ids in input_ids]
            ).to(self.current_device)

        input_data.pop("labels", None)  # we don't want to compute LM losses
        input_data['images'] = images
        return input_data
    def compute_advantages(
        self,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        mask: torch.FloatTensor,
    ):
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1]

        values = values * mask
        rewards = rewards * mask

        if self.config.whiten_rewards:
            rewards = masked_whiten(rewards, mask, shift_mean=False)

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()
        return values, advantages, returns

    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: PreTrainedModelWrapper,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
        response_masks: Optional[torch.Tensor] = None,
    ):
        """
        Calculate model outputs in multiple batches.

        Args:
            queries (`torch.LongTensor`):
                List of tensors containing the encoded queries, shape (`batch_size`, `query_length`)
            responses (`torch.LongTensor`):
                List of tensors containing the encoded responses, shape (`batch_size`, `response_length`)
            return_logits (`bool`, *optional*, defaults to `False`):
                Whether to return all_logits. Set to `False` if logits are not needed to reduce memory consumption.
        Returns:
            (tuple):
                - all_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_ref_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_values (`torch.FloatTensor`): Values of the responses, shape (`batch_size`, `response_length`)
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        model.eval()

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            logits, _, values = model(**input_kwargs)

            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]
            new_masks = torch.ones(values.size(0), values.size(1)).cuda()
            for j in range(len(query_batch)):
                if self.is_encoder_decoder:
                    # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
                    start = 1
                    end = attention_mask[j, :].sum() - 1
                else:
                    start = len(query_batch[j]) - 1  # logprobs starts from the second query token
                    if attention_mask[j, 0] == 0:  # offset left padding
                        start += attention_mask[j, :].nonzero()[0]
                    end = start + len(response_batch[j])
                    if response_masks is not None:
                        response_masks_batch[j] = torch.cat(
                            (torch.zeros_like(query_batch[j]), response_masks_batch[j])
                        )[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0

                new_masks[j, :start+99] = 0
                new_masks[j, end+99: ] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(new_masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @PPODecorators.empty_device_cache()
    def train_minibatch(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
    ):
        """
        Train one PPO minibatch

        Args:
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape [mini_batch_size, response_length]
            values (`torch.FloatTensor`):
                Values of the value head, shape [mini_batch_size, response_length]
            query (`torch.LongTensor`):
                Encoded queries, shape [mini_batch_size, query_length]
            response (`torch.LongTensor`):
                Encoded responses, shape [mini_batch_size, response_length]
            model_input (`torch.LongTensor`):
                Concatenated queries and responses, shape [mini_batch_size, query_length+response_length]

        Returns:
            train_stats (dict[str, `torch.Tensor`]):
                Dictionary of training statistics
        """
        self.model.train()
        loss_p, loss_v, train_stats = self.loss(
            old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns
        )
        loss = loss_p + loss_v
        self.accelerator.backward(loss)
        if self.config.max_grad_norm is not None:
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model_params, self.config.max_grad_norm)
        self.optimizer.step()
        # we call optimizer.zero_grad() every time and let `accelerator` handle accumulation
        # see https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#the-finished-code
        self.optimizer.zero_grad()
        return train_stats

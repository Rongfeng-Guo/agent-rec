# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override
from torch.nn.functional import log_softmax
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_equal_to_4_46
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

       

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma

        #adpo hyperparams
        if self.loss_type == "adpo":
            self.adpo_kl_type = finetuning_args.adpo_kl_type
            self.adpo_kl_weight = finetuning_args.adpo_kl_weight

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.callback_handler.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def get_batch_samples(self, epoch_iterator, num_batches):
        r"""
        Replaces the method of KTO Trainer with the one of the standard Trainer.
        """
        return Trainer.get_batch_samples(self, epoch_iterator, num_batches)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss

    @override
    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        A simple sigmoid-based DPO loss that allows using 'adpo' as loss_type
        without triggering TRL's internal check.
        """
        logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
        losses = -F.logsigmoid(self.beta * logits)

        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def adpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        action_policy_logps: torch.FloatTensor,     # [B, T, V]
        action_reference_logps: torch.FloatTensor,  # [B, T, V]
        action_mask: torch.BoolTensor,              # [B, T, 1]
        kl_type: str = "l2",
        kl_coef: float = 1.0,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        # === Step 1: 计算标准 DPO Loss ===
        dpo_losses, dpo_chosen_rewards, dpo_rejected_rewards = self.dpo_loss(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps
        )

        # === Step 2: 计算 Action KL Loss（masked） ===
        if kl_type == "l2":
            diff = (action_policy_logs - action_reference_logs) ** 2
        elif kl_type == "abs":
            diff = torch.abs(action_policy_logps - action_reference_logps)
        else:
            raise ValueError(f"Unsupported kl_type: {kl_type}")

        # 应用 mask，只在 action 区域计算
        masked_diff = diff * action_mask  # [B, T, V]
        vocab_size = diff.size(-1)
        token_count = action_mask.sum(dim=(1, 2)).clamp(min=1)  # 防止除以0

        kl_reg = masked_diff.sum(dim=(1, 2)) / (token_count * vocab_size)

        # === Step 3: 加权合并 ===
        total_loss = dpo_losses + kl_coef * kl_reg

        return total_loss, dpo_chosen_rewards, dpo_rejected_rewards
   
    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
        action_policy_logps: Optional["torch.Tensor"] = None,
        action_reference_logps: Optional["torch.Tensor"] = None,
        action_mask: Optional["torch.Tensor"] = None,  # <--- 新增
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            if self.loss_type == "adpo":
                losses, chosen_rewards, rejected_rewards = self.adpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    reference_chosen_logps, reference_rejected_logps,
                    action_policy_logps, action_reference_logps,
                    action_mask=action_mask,  # <--- 新增
                    kl_type=self.finetuning_args.adpo_kl_type,
                    kl_coef=self.finetuning_args.adpo_kl_weight,
                )
            else:
                losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
                )

        return losses, chosen_rewards, rejected_rewards

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.finetuning_args.use_ref_model:
            batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error

        all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length

    # @override
    # def compute_reference_log_probs(
    #     self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    # ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
    #     r"""
    #     Computes log probabilities of the reference model.
    #     """
    #     if not self.finetuning_args.use_ref_model:
    #         return None, None

    #     if self.ref_model is None:
    #         ref_model = model
    #         ref_context = self.accelerator.unwrap_model(model).disable_adapter()
    #     else:
    #         ref_model = self.ref_model
    #         ref_context = nullcontext()

    #     with torch.no_grad(), ref_context:
    #         reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)

    #     return reference_chosen_logps, reference_rejected_logps


    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Computes log probabilities of the reference model, and optionally its logits (if ADPO).
        """
        if not self.finetuning_args.use_ref_model:
            return None, None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            # 修改点：返回 logits
            ref_chosen_logps, ref_rejected_logps, ref_chosen_logits, ref_rejected_logits, _ = self.concatenated_forward(ref_model, batch)

        return ref_chosen_logps, ref_rejected_logps, ref_chosen_logits

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        metrics = {}

        # if self.loss_type == "adpo":
        #     labels = batch["labels"]
        #     for i in range(labels.size(0)):
        #         non_masked = (labels[i] != -100).nonzero(as_tuple=True)[0]
        #         if len(non_masked) > 0:
        #             first = non_masked[0].item()
        #             labels[i, first] = -100

        # 比如看两个模型的一个参数：
        
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, batch)

        # 2. reference forward（注意修改成返回 logits）
        reference_chosen_logps, reference_rejected_logps, reference_chosen_logits = self.compute_reference_log_probs(model, batch)
        # labels = batch["labels"]
        # num_total = labels.numel()
        # num_masked = (labels == -100).sum().item()
        # num_valid = num_total - num_masked
        # print(f"🔢 Total tokens: {num_total}")
        # print(f"❌ -100 (masked) tokens: {num_masked}")
        # print(f"✅ Valid (non-masked) tokens: {num_valid}")
        # print(f"🧮 Masked ratio: {num_masked / num_total:.2%}")

        
        # print("labels", batch["labels"])
        # print("labels shape", batch["labels"].shape)
        
        # print("reference_chosen_logits")
        # print(reference_chosen_logits) 
        # print(reference_chosen_logits.shape)
        # 6.25 ADPO_V6 用mask代替kl 不太行，还是只会ask

                  

        # 3. ADPO: 处理 prefix KL
        if self.loss_type == "adpo":
            input_ids = batch["input_ids"]
            batch_size = input_ids.shape[0] // 2
            chosen_ids, rejected_ids = input_ids[:batch_size], input_ids[batch_size:]
            labels = batch["labels"]
            chosen_labels = labels[:batch_size]

            # 当前：prompt + action 的总长度
            prefix_lens = (chosen_ids == rejected_ids).cumprod(dim=1).sum(dim=1)

            # max prefix_len across the batch (because pad_sequence)
            max_prefix_len = prefix_lens.max().item()

            # 构造 token 位置信息
            arange_ids = torch.arange(max_prefix_len, device=prefix_lens.device).unsqueeze(0)  # [1, max_prefix_len]


            # 利用 labels 找到 output 起点（第一个非 -100 的位置）
            output_start = (chosen_labels != -100).float().argmax(dim=1)

            # 精确的 action 区域 = prompt + action - prompt = prefix_len - output_start
            action_lens = prefix_lens - output_start
            action_lens = action_lens.clamp(min=0)  # 防止负值

            # 构造 mask，保留位置在 action 区域的
            # output_start, action_lens 都是 [batch]
            action_end = output_start + action_lens
            action_mask = (arange_ids >= output_start.unsqueeze(1)) & (arange_ids < action_end.unsqueeze(1))  # [B, T]
            action_mask = action_mask.unsqueeze(-1)  # [B, T, 1]
            # print("chosen_ids")
            # print(chosen_ids)
            # print(chosen_ids.shape)
            # print("rejected_ids")
            # print(rejected_ids)
            # print(rejected_ids.shape)
            # print("prefix_lens")
            # print(prefix_lens)
            # print("action_lens")
            # print(action_lens)
            # print("action_mask")
            # print(action_mask)
            # print(action_mask.shape)
            action_policy_logps = []
            action_reference_logps = []

            for i in range(batch_size):
                l = prefix_lens[i].item()
                if l == 0:
                    raise ValueError(f"Sample {i} has zero prefix length")

                policy_prefix_logits = policy_chosen_logits[i, :l]
                policy_logps = log_softmax(policy_prefix_logits, dim=-1)
                action_policy_logps.append(policy_logps)

                reference_prefix_logits = reference_chosen_logits[i, :l]
                reference_logps = log_softmax(reference_prefix_logits, dim=-1)
                action_reference_logps.append(reference_logps)

            # pad
            action_policy_logps = torch.nn.utils.rnn.pad_sequence(action_policy_logps, batch_first=True)
            action_reference_logps = torch.nn.utils.rnn.pad_sequence(action_reference_logps, batch_first=True)

            if action_policy_logps.shape != action_reference_logps.shape:
                raise ValueError(f"Prefix logps shape mismatch: {action_policy_logps.shape} vs {action_reference_logps.shape}")

            losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                action_policy_logps,
                action_reference_logps,
                action_mask=action_mask  # 👈 新增
            )

            # === SFT loss with prefix mask ===
            policy_logps = log_softmax(policy_chosen_logits, dim=-1)
            input_ids = batch["input_ids"][:batch_size]
            chosen_logps = policy_logps.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

            seq_len = chosen_logps.shape[1]
            sft_mask = torch.arange(seq_len, device=chosen_logps.device)[None, :] >= prefix_lens[:, None]
            sft_loss = - (chosen_logps * sft_mask).sum() / sft_mask.sum().clamp(min=1)

        else:
            losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
            sft_loss = -policy_chosen_logps_avg

        # === SFT 正则加权 ===
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        # 5. Metrics logging
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()
        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
            metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / self.beta).mean().item()

        return losses.mean(), metrics


    # @override
    # def get_batch_loss_metrics(
    #     self,
    #     model: "PreTrainedModel",
    #     batch: Dict[str, "torch.Tensor"],
    #     train_eval: Literal["train", "eval"] = "train",
    # ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
    #     metrics = {}

    #     # 1. policy forward
    #     (
    #         policy_chosen_logps,
    #         policy_rejected_logps,
    #         policy_chosen_logits,
    #         policy_rejected_logits,
    #         policy_chosen_logps_avg,
    #     ) = self.concatenated_forward(model, batch)

    #     # 2. reference forward（注意修改成返回 logits）
    #     reference_chosen_logps, reference_rejected_logps, reference_chosen_logits = self.compute_reference_log_probs(model, batch)

    #     # 3. ADPO: 处理 prefix KL
    #     if self.loss_type == "adpo":
    #         input_ids = batch["input_ids"]
    #         batch_size = input_ids.shape[0] // 2
    #         chosen_ids, rejected_ids = input_ids[:batch_size], input_ids[batch_size:]
    #         prefix_lens = (chosen_ids == rejected_ids).cumprod(dim=1).sum(dim=1)  # [batch]

    #         action_policy_logps = []
    #         action_reference_logps = []

    #         for i in range(batch_size):
    #             l = prefix_lens[i].item()
    #             if l == 0:
    #                 raise ValueError(f"Sample {i} has zero prefix length")

    #             policy_prefix_logits = policy_chosen_logits[i, :l]
    #             policy_logps = log_softmax(policy_prefix_logits, dim=-1)
    #             action_policy_logps.append(policy_logps)

    #             reference_prefix_logits = reference_chosen_logits[i, :l]
    #             reference_logps = log_softmax(reference_prefix_logits, dim=-1)
    #             action_reference_logps.append(reference_logps)

    #         # pad
    #         action_policy_logps = torch.nn.utils.rnn.pad_sequence(action_policy_logps, batch_first=True)
    #         action_reference_logps = torch.nn.utils.rnn.pad_sequence(action_reference_logps, batch_first=True)

    #         if action_policy_logps.shape != action_reference_logps.shape:
    #             raise ValueError(f"Prefix logps shape mismatch: {action_policy_logps.shape} vs {action_reference_logps.shape}")

    #         losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
    #             policy_chosen_logps,
    #             policy_rejected_logps,
    #             reference_chosen_logps,
    #             reference_rejected_logps,
    #             action_policy_logps,
    #             action_reference_logps,
    #         )
    #     else:
    #         losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
    #             policy_chosen_logps,
    #             policy_rejected_logps,
    #             reference_chosen_logps,
    #             reference_rejected_logps,
    #         )

    #     # 4. SFT regularization
    #     sft_loss = -policy_chosen_logps_avg
    #     if self.ftx_gamma > 1e-6:
    #         losses += self.ftx_gamma * sft_loss

    #     # 5. Metrics logging
    #     prefix = "eval_" if train_eval == "eval" else ""
    #     metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
    #     metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
    #     metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
    #     metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
    #     metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
    #     metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
    #     metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
    #     metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()
    #     if self.loss_type == "orpo":
    #         metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
    #         metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / self.beta).mean().item()

    #     return losses.mean(), metrics


    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r"""
        Fixes the loss value for transformers 4.46.0.
        https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/trainer.py#L3605
        """
        loss = super().compute_loss(model, inputs, return_outputs)
        if is_transformers_version_equal_to_4_46() and kwargs.pop("num_items_in_batch", False):
            if return_outputs:
                return (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
            else:
                return loss / self.args.gradient_accumulation_steps

        return loss

    @override
    def log(self, logs: Dict[str, float]) -> None:
        r"""
        Log `logs` on the various objects watching training, including stored metrics.
        """
        # logs either has "loss" or "eval_loss"
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        key_list, metric_list = [], []
        for key, metrics in self._stored_metrics[train_eval].items():
            key_list.append(key)
            metric_list.append(torch.tensor(metrics, dtype=torch.float).to(self.accelerator.device).mean().item())

        del self._stored_metrics[train_eval]
        if len(metric_list) < 10:  # pad to for all reduce
            for i in range(10 - len(metric_list)):
                key_list.append(f"dummy_{i}")
                metric_list.append(0.0)

        metric_list = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
        metric_list = self.accelerator.reduce(metric_list, "mean").tolist()
        for key, metric in zip(key_list, metric_list):  # add remaining items
            if not key.startswith("dummy_"):
                logs[key] = metric

        return Trainer.log(self, logs)

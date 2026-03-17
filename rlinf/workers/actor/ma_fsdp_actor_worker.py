import os
import sys
import time
import copy
from functools import partial
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    kl_penalty,
)
from rlinf.config import SupportedModel, torch_dtype_from_precision
from rlinf.data.io_struct import BatchResizingIterator, RolloutResult, DynamicRolloutResult
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.hybrid_engines.fsdp.utils import (
    pack_fsdp,
    prepare_pack_fsdp,
    prepare_pack_fsdp_right_padding,
    unpack_fsdp,
    pack_sequences,
    unpack_sequences,
)
from rlinf.models import get_model
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
    split_dynamic_batch_size,
)
from rlinf.utils.distributed import (
    RolloutDataBalance,
    all_reduce_dict,
    masked_normalization,
)
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    cat_list_of_dict_tensor,
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    cpu_weight_swap,
    get_loss_agg_func,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
)
from rlinf.workers.rollout.utils import RankMapper

from rlinf.workers.actor.fsdp_actor_worker import FSDPActor

class MAFSDPActor(FSDPActor):
    def __init__(
        self, cfg: DictConfig, placement: ModelParallelComponentPlacement, role='actor'
    ):
        super().__init__(cfg, placement)
        self.role = role

        print(f'MA FSDP Actor is here', file=sys.stderr)

        # self.is_dynamic_rollout_batch = self.cfg.agentloop.is_dynamic_rollout_batch
        # assert self.is_dynamic_rollout_batch
        # assert self.enable_dp_load_balance, "enable_dp_load_balance must be True when is_dynamic_rollout_batch is True"
        # assert self.placement_mode == PlacementMode.COLLOCATED

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], DynamicRolloutResult]:

        result: DynamicRolloutResult = channel.get()

        batch = result.to_actor_batch(
            # self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )

        return batch, result

    def training_step(
        self, batch: dict[str, torch.Tensor] | BatchResizingIterator
    ) -> tuple[dict[str, torch.Tensor], float, list[float]]:
        if isinstance(batch, dict):
            global_batch_size = batch["input_ids"].shape[0]
            assert global_batch_size % self.micro_batch_size == 0, (
                f"global batch size {global_batch_size} can not divide micro_batch_size {self.micro_batch_size}"
            )
            micro_batches_iter, micro_batch_cnt, _ = self._split_to_micro_batch(
                batch,
                self.enable_dynamic_batch_size,
                max_tokens_per_mbs=self.max_tokens_per_mbs,
                split_num=global_batch_size // self.micro_batch_size,
            )
            self.gradient_accumulation = micro_batch_cnt
        else:
            global_batch_size = self.total_batch_size_per_dp // self.n_mini_batches
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt

            def iterator_wrapper():
                for _ in range(micro_batch_cnt):
                    yield next(batch)

            micro_batches_iter = iterator_wrapper()
        self.optimizer.zero_grad()
        mbs_metrics_list = {}
        for idx, m_batch in enumerate(micro_batches_iter):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == micro_batch_cnt,
            )
            for k, v in m_batch.items():
                m_batch[k] = v.cuda() if isinstance(v, torch.Tensor) else v

            multi_modal_inputs = {}
            if "multi_modal_inputs" in m_batch.keys():
                for key in m_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in m_batch["multi_modal_inputs"]],
                        dim=0,
                    ).cuda()

            input_ids = m_batch["input_ids"]
            attention_mask = m_batch["attention_mask"]
            position_ids = m_batch["position_ids"]
            response_mask = m_batch["response_mask"]

            ref_logprobs = None
            if "ref_logprobs" in m_batch:
                ref_logprobs = m_batch["ref_logprobs"]
            

            label = copy.deepcopy(input_ids)
            label[:, -1] = 0
            label[:, :-1] = input_ids[:, 1:]

            # breakpoint()
            # loss_mask[:, :-1] = response_mask[:, 1:]
            # 此处loss_mask是否应该shift存疑
            
            if self.enable_dynamic_batch_size:
                max_seq_len_pack = self.max_tokens_per_mbs
                max_seq_len_unpack = self.cfg.actor.model.encoder_seq_length
                max_prompt_len = self.cfg.data.max_prompt_length
                max_response_len = max_seq_len_unpack - max_prompt_len
                idx_starts, idx_ends = prepare_pack_fsdp_right_padding(m_batch)
                input_ids, position_ids, attention_mask = pack_fsdp(
                    input_ids,
                    position_ids,
                    idx_starts=idx_starts,
                    idx_ends=idx_ends,
                    max_seq_len_pack=max_seq_len_pack,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                label = pack_sequences(
                    label, idx_starts, idx_ends, max_seq_len_pack, self.tokenizer.eos_token_id,
                ).unsqueeze(0)
                # loss_mask = pack_sequences(
                #     loss_mask, idx_starts, idx_ends, max_seq_len_pack, self.tokenizer.eos_token_id,
                # ).unsqueeze(0)

            with self.amp_context:
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )

            logits: torch.Tensor = output.logits
            logits.div_(self.cfg.algorithm.sampling_params.temperature)
            if self.enable_dynamic_batch_size:

                logprobs = compute_logprobs_from_logits(logits, label, op_type=self.entropy_op_type)
                logprobs = F.pad(logprobs[:, :-1], (1, 0)) # right shift logprobs                pass
                logprobs = unpack_sequences(logprobs, idx_starts, idx_ends, max_seq_len_unpack, pad_val=0)

            else:

                logprobs = compute_logprobs_from_logits(
                    logits, label, op_type=self.entropy_op_type
                )
                logprobs = F.pad(logprobs[:, :-1], (1, 0)) # assume shape is [B, T]

                # 此处没有像单轮情况下一样先右移logits，再用logits和input ids求logprobs
                # 而是通过label（label就是左移过的input_ids）和logits先求logprobs，再把logprobs右移
            logprobs *= response_mask

            task = 'sft' # or 'rl'

            if task == 'sft':
                ce_loss = self.loss_agg_func(-logprobs, mask=response_mask)
                ce_loss = ce_loss / self.gradient_accumulation
                with backward_ctx:
                    self.grad_scaler.scale(ce_loss).backward()

                mbs_metrics_data = {}
                mbs_metrics_data.update(
                    {
                        "actor/ce_loss": ce_loss.detach(),
                    }
                )
                
                append_to_dict(mbs_metrics_list, mbs_metrics_data)
                
            if task == 'rl':
                clip_ratio = self.cfg.algorithm.ratio_clip_eps
                clip_ratio_low = self.cfg.algorithm.get("clip_ratio_low", None)
                clip_ratio_high = self.cfg.algorithm.get("clip_ratio_high", None)
                clip_ratio_low = (
                    clip_ratio_low if clip_ratio_low is not None else clip_ratio
                )
                clip_ratio_high = (
                    clip_ratio_high if clip_ratio_high is not None else clip_ratio
                )
                clip_ratio_c = self.cfg.algorithm.get("clip_ratio_c", 3.0)

                advantages = m_batch["advantages"]
                prev_logprobs = m_batch["prev_logprobs"]
                if self.cfg.algorithm.get("importance_sampling_fix", False):
                    rollout_prev_logprobs = prev_logprobs
                    recompute_prev_logprobs = m_batch["recompute_prev_logprobs"]
                    advantages = advantages * torch.clamp(
                        (recompute_prev_logprobs - rollout_prev_logprobs).exp(),
                        min=self.cfg.algorithm.importance_sampling_clip,
                    )
                
                loss, mbs_metrics_data = policy_loss(
                    loss_type=self.cfg.algorithm.loss_type,
                    loss_agg_func=self.loss_agg_func,
                    logprobs=logprobs,
                    old_logprobs=prev_logprobs,
                    advantages=advantages,
                    clip_ratio_low=clip_ratio_low,
                    clip_ratio_high=clip_ratio_high,
                    clip_ratio_c=clip_ratio_c,
                    loss_mask=response_mask,
                    task_type=self.task_type,
                )
                
                entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                if self.calculate_entropy:
                    entropy = compute_entropy_from_logits(
                        logits,
                    )
                    entropy = F.pad(entropy[:, :-1], (1, 0)) # right shift
                    if self.enable_dynamic_batch_size:
                        entropy = unpack_sequences(entropy, idx_starts, idx_ends, max_seq_len_unpack, pad_val=0)

                    entropy_loss = self.loss_agg_func(entropy, mask=response_mask)
                    if self.calculate_entropy_loss:
                        loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss
                
                kl_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                if self.kl_beta > 0 and ref_logprobs is not None:
                    # 所以loss mask确实不应该左移
                    kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
                    kl_loss = self.loss_agg_func(kld, response_mask)
                    loss = loss + kl_loss * self.kl_beta
                
                # add to log
                # scale loss for gradient accumulation and backprop
                final_loss_metric = loss.detach()
                loss = loss / self.gradient_accumulation
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()
                
                mbs_metrics_data.update(
                    {
                        "actor/final_loss": final_loss_metric,
                        "actor/entropy_loss": entropy_loss.detach(),
                        "actor/kl_loss": kl_loss.detach(),
                    }
                )
                
                append_to_dict(mbs_metrics_list, mbs_metrics_data)

        grad_norm, lr_list = self.optimizer_step()

        # put lr scheduler step here
        self.lr_scheduler.step()

        # aggregate metrics across micro-batches
        mean_metric_dict = {
            key: torch.mean(torch.stack(value))
            for key, value in mbs_metrics_list.items()
        }
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        mean_metric_dict["actor/grad_norm"] = float(grad_norm)
        mean_metric_dict["actor/lr"] = lr_list[0]
        return mean_metric_dict

    @torch.no_grad()
    def inference_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        response_mask = batch["response_mask"]
        position_ids = batch["position_ids"]

        label = copy.deepcopy(input_ids)
        label[:, -1] = 0
        label[:, :-1] = input_ids[:, 1:]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in batch.keys():
            for key in batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in batch["multi_modal_inputs"]],
                    dim=0,
                ).cuda()

        if self.enable_dynamic_batch_size:
            max_seq_len_pack = self.max_tokens_per_mbs
            max_seq_len_unpack = self.cfg.actor.model.encoder_seq_length
            max_prompt_len = self.cfg.data.max_prompt_length
            max_response_len = max_seq_len_unpack - max_prompt_len
            idx_starts, idx_ends = prepare_pack_fsdp_left_padding(batch)

            input_ids, position_ids, attention_mask = pack_fsdp(
                input_ids,
                position_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_pack=max_seq_len_pack,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            label = pack_sequences(
                label, idx_starts, idx_ends, max_seq_len_pack, self.tokenizer.eos_token_id,
            ).unsqueeze(0)

        with self.amp_context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                **multi_modal_inputs,
            )

        logits: torch.Tensor = outputs.logits
        logits = logits / self.cfg.algorithm.sampling_params.temperature
        if self.enable_dynamic_batch_size:

            logprobs = compute_logprobs_from_logits(logits, label, op_type=self.entropy_op_type)
            logprobs = F.pad(logprobs[:, :-1], (1, 0)) # right shift logprobs
            logprobs = unpack_sequences(logprobs, idx_starts, idx_ends, max_seq_len_unpack, pad_val=0)
            
        else:
            logprobs = compute_logprobs_from_logits(
                logits, label, op_type=self.entropy_op_type
            )
            logprobs = F.pad(logprobs[:, :-1], (1, 0)) # assume shape is [B, T]

        logprobs *= response_mask

        return logprobs

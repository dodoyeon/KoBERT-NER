# trainer utils
import torch.distributed as dist


def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

# replay buffer utils
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from chatgpt.experience_maker.base import Experience


@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    values: (1)
    reward: (1)
    advatanges: (1)
    attention_mask: (S)
    action_mask: (A)

    "A" is the number of actions.
    """
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    reward: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]


def split_experience_batch(experience: Experience) -> List[BufferItem]:
    batch_size = experience.sequences.size(0)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = ('sequences', 'action_log_probs', 'values', 'reward', 'advantages', 'attention_mask', 'action_mask')
    for key in keys:
        value = getattr(experience, key)
        if isinstance(value, torch.Tensor):
            vals = torch.unbind(value)
        else:
            # None
            vals = [value for _ in range(batch_size)]
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v
    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = 'left') -> torch.Tensor:
    assert side in ('left', 'right')
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == 'left' else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def make_experience_batch(items: List[BufferItem]) -> Experience:
    kwargs = {}
    to_pad_keys = set(('action_log_probs', 'action_mask'))
    keys = ('sequences', 'action_log_probs', 'values', 'reward', 'advantages', 'attention_mask', 'action_mask')
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if key in to_pad_keys:
            batch_data = zero_pad_sequences(vals)
        else:
            batch_data = torch.stack(vals, dim=0)
        kwargs[key] = batch_data
    return Experience(**kwargs)

# model utils -> used in loss
from typing import Optional, Union

import loralib as lora
import torch.nn as nn

def compute_approx_kl(log_probs: torch.Tensor,
                      log_probs_base: torch.Tensor,
                      action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs - log_probs_base
    approx_kl = (log_ratio.exp() - 1) - log_ratio
    if action_mask is not None:
        approx_kl = masked_mean(approx_kl, action_mask, dim=1)
        return approx_kl
    approx_kl = approx_kl.mean(dim=1)
    return approx_kl


def compute_reward(r: Union[torch.Tensor, float],
                   kl_coef: float,
                   log_probs: torch.Tensor,
                   log_probs_base: torch.Tensor,
                   action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if kl_coef <= 0.0:
        return r
    kl = compute_approx_kl(log_probs, log_probs_base, action_mask=action_mask)
    reward = r - kl_coef * kl
    return reward


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    tensor = tensor * mask
    tensor = tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim)
    mean = tensor / (mask_sum + 1e-8)
    return mean


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


def normalize(tensor: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    mean = tensor.mean(dim)
    mean_centered = tensor - mean
    var = (mean_centered**2).mean(dim)
    norm = mean_centered * var.clamp(min=eps).rsqrt()
    return norm


def convert_to_lora(model: nn.Module,
                    input_size: int,
                    output_size: int,
                    lora_rank: int = 16,
                    lora_alpha: int = 1,
                    lora_dropout: float = 0.,
                    fan_in_fan_out: bool = False,
                    merge_weights: bool = True):
    if lora_rank > min(input_size, output_size):
        raise ValueError(f"LoRA rank {lora_rank} must be less or equal than {min(input_size, output_size)}")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._modules[name] = lora.Linear(input_size,
                                                output_size,
                                                r=lora_rank,
                                                lora_alpha=lora_alpha,
                                                lora_dropout=lora_dropout,
                                                fan_in_fan_out=fan_in_fan_out,
                                                merge_weights=merge_weights)

# generation_utils
def update_model_kwargs_fn(outputs: dict, **model_kwargs) -> dict:
    if "past_key_values" in outputs:
        model_kwargs["past"] = outputs["past_key_values"]
    else:
        model_kwargs["past"] = None

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    # update attention mask
    if "attention_mask" in model_kwargs:
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

    return model_kwargs
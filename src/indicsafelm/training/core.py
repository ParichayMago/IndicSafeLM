"""Training primitives shared by pretraining, SFT, reward and PPO stages.

``MixedPrecisionTrainer`` keeps fp32 *master* weights in the optimizer and runs
forward/backward on a bf16 *compute* copy ("O2" mixed precision). Apple GPUs
run bf16 matmuls ~6x faster than fp32 but per-op autocast is slow there, so a
whole-model bf16 copy is used instead. With ``precision="fp32"`` the compute
model is the master model itself.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class OptimConfig:
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    total_steps: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0


def lr_at(step: int, cfg: OptimConfig) -> float:
    """Linear warm-up then cosine decay to ``min_lr_ratio * lr``."""

    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps))
    floor = cfg.lr * cfg.min_lr_ratio
    return floor + 0.5 * (cfg.lr - floor) * (1 + math.cos(math.pi * progress))


def _param_groups(model: torch.nn.Module, weight_decay: float) -> List[dict]:
    decay, no_decay, seen = [], [], set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        (decay if param.dim() >= 2 else no_decay).append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


class MixedPrecisionTrainer:
    def __init__(self, master: torch.nn.Module, cfg: OptimConfig, precision: str = "bf16") -> None:
        self.master = master
        self.cfg = cfg
        self.precision = precision
        if precision == "bf16":
            self.compute = copy.deepcopy(master).to(torch.bfloat16)
        elif precision == "fp32":
            self.compute = master
        else:
            raise ValueError(f"unsupported precision {precision}")
        self.master_params = self._unique(master.parameters())
        self.compute_params = self._unique(self.compute.parameters())
        assert len(self.master_params) == len(self.compute_params)
        self.optimizer = torch.optim.AdamW(
            _param_groups(master, cfg.weight_decay), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps
        )

    @staticmethod
    def _unique(params: Iterable[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
        seen, out = set(), []
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        return out

    def train(self, mode: bool = True) -> None:
        self.master.train(mode)
        self.compute.train(mode)

    def step(self, step_index: int) -> dict:
        """Move compute gradients to the master copy, clip, update and re-sync."""

        if self.compute is not self.master:
            for pm, pc in zip(self.master_params, self.compute_params):
                pm.grad = None if pc.grad is None else pc.grad.detach().float()
                pc.grad = None
        grads = [p.grad for p in self.master_params if p.grad is not None]
        finite = all(torch.isfinite(g).all() for g in grads)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params, self.cfg.grad_clip) if grads else torch.tensor(0.0)
        lr = lr_at(step_index, self.cfg)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        if finite:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.sync()
        return {"lr": lr, "grad_norm": float(grad_norm), "finite": bool(finite)}

    @torch.no_grad()
    def sync(self) -> None:
        if self.compute is not self.master:
            for pm, pc in zip(self.master_params, self.compute_params):
                pc.copy_(pm)

    def state_dict(self) -> dict:
        return {"master": self.master.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.master.load_state_dict(state["master"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.sync()


def masked_token_nll(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> tuple:
    """Return (sum of NLL over masked targets, number of masked targets).

    ``logits`` (B, T, V) predict ``targets`` (B, T); ``mask`` (B, T) selects the
    positions that count (padding and, for SFT, the prompt are excluded). The
    caller divides the two numbers, so the denominator is always explicit
    (FR-M02, Appendix B.3).
    """

    per_token = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    mask = mask.reshape(-1).to(per_token.dtype)
    return (per_token * mask).sum(), mask.sum()


def token_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Log-probability of each target token (B, T) under ``logits`` (B, T, V)."""

    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


def right_pad(sequences: List[List[int]], pad_id: int, device: Optional[torch.device] = None) -> tuple:
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, seq in enumerate(sequences):
        ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[row, : len(seq)] = 1
    if device is not None:
        ids, mask = ids.to(device), mask.to(device)
    return ids, mask

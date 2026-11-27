"""Compact decoder-only Transformer used for every IndicSafeLM checkpoint.

Design (report section 6.1, Table 6.1):

* token embedding -> N pre-norm decoder blocks -> final LayerNorm -> vocabulary
  projection whose weight is tied to the embedding;
* causal multi-head self-attention with rotary position embeddings (RoPE)
  applied to queries and keys;
* GELU feed-forward network; dropout only during pretraining.

The same module serves the base, SFT and PPO policies. The reward and value
models wrap the backbone with a scalar head (see ``ScalarHeadModel``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    n_layer: int = 8
    d_model: int = 512
    n_head: int = 8
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.10
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    pad_id: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def rope_tables(head_dim: int, max_len: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cos/sin tables of shape (max_len, head_dim // 2)."""

    if head_dim % 2:
        raise ValueError("RoPE needs an even head dimension")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate feature pairs of ``x`` (B, H, T, D) by position-dependent angles.

    Uses the "rotate half" pairing (feature i with feature i + D/2), which is
    equivalent to RoFormer's interleaved pairing up to a fixed permutation of
    the head features and maps to contiguous slices on the GPU.

    ``cos``/``sin`` have shape (B, 1, T, D/2) so every sequence in a batch can use
    its own positions (needed for left-padded generation).
    """

    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_head:
            raise ValueError("d_model must be divisible by n_head")
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bsz, seq, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=2)
        q = q.view(bsz, seq, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        # Dropout is applied on the residual branches (DecoderBlock), not inside
        # attention: the fused MPS attention kernel has no dropout support.
        if attn_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(bsz, seq, width)
        return self.proj(out), (k, v)


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, attn_mask, past=None):
        attn_out, present = self.attn(self.norm1(x), cos, sin, attn_mask, past)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, present


class TransformerLM(nn.Module):
    """Decoder-only language model with tied input/output embeddings."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.n_layer))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_tables(cfg.d_model // cfg.n_head, cfg.max_seq_len * 2, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # GPT-2 style scaled init of residual output projections.
        for name, param in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("ff.down.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        """Unique trainable parameters (tied weights counted once)."""

        seen = {}
        for param in self.parameters():
            seen[id(param)] = param.numel()
        return sum(seen.values())

    def hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[KVCache] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """Run the decoder stack and return final-normalised hidden states.

        ``attention_mask`` (B, past+T) marks valid (non-padding) key positions. It
        is only needed for padded batches; packed pretraining uses the plain
        causal fast path.
        """

        bsz, seq = input_ids.shape
        past_len = 0 if past_kv is None else past_kv[0][0].shape[2]
        if position_ids is None:
            if attention_mask is not None:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)[:, past_len:]
            else:
                position_ids = torch.arange(past_len, past_len + seq, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(bsz, seq)
        if int(position_ids.max()) >= self.rope_cos.shape[0]:
            raise ValueError("sequence longer than the supported context")
        cos = self.rope_cos[position_ids].unsqueeze(1)
        sin = self.rope_sin[position_ids].unsqueeze(1)

        attn_mask = None
        if attention_mask is not None or past_kv is not None:
            total = past_len + seq
            key_valid = (
                attention_mask.bool()
                if attention_mask is not None
                else torch.ones(bsz, total, dtype=torch.bool, device=input_ids.device)
            )
            query_pos = torch.arange(past_len, total, device=input_ids.device)
            key_pos = torch.arange(total, device=input_ids.device)
            causal = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)  # (T, total)
            attn_mask = causal.unsqueeze(0) & key_valid[:, None, :]
            # A padding query with no valid key would produce NaNs; let it see itself.
            diag = key_pos.unsqueeze(0) == query_pos.unsqueeze(1)
            attn_mask = (attn_mask | diag.unsqueeze(0)).unsqueeze(1)  # (B, 1, T, total)

        x = self.drop(self.embed(input_ids))
        presents: KVCache = []
        for i, block in enumerate(self.blocks):
            past = None if past_kv is None else past_kv[i]
            x, present = block(x, cos, sin, attn_mask, past)
            if use_cache:
                presents.append(present)
        return self.norm(x), (presents if use_cache else None)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[KVCache] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ):
        hidden, presents = self.hidden_states(input_ids, attention_mask, past_kv, position_ids, use_cache)
        logits = self.lm_head(hidden)
        return (logits, presents) if use_cache else logits


class ScalarHeadModel(nn.Module):
    """Backbone plus a scalar head; used for the reward model and PPO critic."""

    def __init__(self, backbone: TransformerLM) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.cfg.d_model, 1)
        nn.init.normal_(self.head.weight, std=1.0 / math.sqrt(backbone.cfg.d_model + 1))
        nn.init.zeros_(self.head.bias)

    def token_values(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.backbone.hidden_states(input_ids, attention_mask)
        return self.head(hidden).squeeze(-1)

    def score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Scalar per sequence read at the last valid (non-padding) position.

        Works for both right padding and left padding: the last valid index is
        the highest position whose mask is 1.
        """

        values = self.token_values(input_ids, attention_mask)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        last = (attention_mask.long() * positions).argmax(dim=1)
        return values.gather(1, last.unsqueeze(1)).squeeze(1)

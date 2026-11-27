"""Shared inference layer: chat template, batched decoding and status codes.

Every stage (evaluation, PPO rollouts, demo) builds prompts and decodes through
this module so the same template, limits and stop rules apply everywhere
(Table 4.3, section 8.1).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from ..tokenizer import TokenizerBundle

STATUS_OK = "ok"
STATUS_TRUNCATED = "truncated"       # hit max_new_tokens without </s>
STATUS_EMPTY = "empty"               # finished with no visible text
STATUS_ROLE_MARKER = "role_marker"   # model emitted <|user|> (template failure)
STATUS_PROMPT_TOO_LONG = "prompt_too_long"
STATUS_ERROR = "error"


@dataclass
class DecodingConfig:
    max_new_tokens: int = 96
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    seed: int = 1735
    max_prompt_tokens: int = 256

    def to_dict(self) -> dict:
        return asdict(self)


def chat_prompt_ids(tok: TokenizerBundle, prompt: str) -> List[int]:
    """``<s> <|user|> prompt <|assistant|>`` — the instruction template."""

    return [tok.bos_id, tok.user_id, *tok.encode(prompt), tok.assistant_id]


def continuation_prompt_ids(tok: TokenizerBundle, prompt: str) -> List[int]:
    """``<s> prompt`` — how the base model is queried (plain continuation)."""

    return [tok.bos_id, *tok.encode(prompt)]


def chat_example(tok: TokenizerBundle, prompt: str, response: str) -> tuple:
    """Token IDs and assistant-span mask for one SFT example.

    The mask covers the response tokens and the closing ``</s>``; the prompt is
    excluded from the supervised loss (section 6.6).
    """

    prompt_ids = chat_prompt_ids(tok, prompt)
    response_ids = [*tok.encode(response), tok.eos_id]
    ids = prompt_ids + response_ids
    mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    return ids, mask


def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        remove = probs.cumsum(-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    return logits


@dataclass
class Generation:
    prompt_ids: List[int]
    response_ids: List[int]
    status: str
    text: str = ""
    seconds: float = 0.0
    extra: Dict = field(default_factory=dict)


@torch.no_grad()
def generate_batch(
    model: torch.nn.Module,
    tok: TokenizerBundle,
    prompts: Sequence[List[int]],
    cfg: DecodingConfig,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> List[Generation]:
    """Left-padded batched sampling with a KV cache.

    Sampling happens on the CPU with an explicit ``torch.Generator`` so a fixed
    seed reproduces the same draws on any accelerator. ``temperature == 0``
    means greedy decoding.
    """

    started = time.time()
    model.eval()
    if generator is None:
        generator = torch.Generator().manual_seed(cfg.seed)
    width = max(len(p) for p in prompts)
    bsz = len(prompts)
    ids = torch.full((bsz, width), tok.pad_id, dtype=torch.long)
    mask = torch.zeros((bsz, width), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        ids[row, width - len(prompt):] = torch.tensor(prompt)
        mask[row, width - len(prompt):] = 1
    ids, mask = ids.to(device), mask.to(device)

    logits, past = model(ids, attention_mask=mask, use_cache=True)
    next_logits = logits[:, -1, :]
    outputs: List[List[int]] = [[] for _ in range(bsz)]
    done = [False] * bsz
    status = [STATUS_TRUNCATED] * bsz
    stop = {tok.eos_id: STATUS_OK, tok.user_id: STATUS_ROLE_MARKER}
    for _ in range(cfg.max_new_tokens):
        step_logits = next_logits.float().cpu()
        if cfg.temperature <= 0:
            chosen = step_logits.argmax(-1)
        else:
            filtered = _filter_logits(step_logits / cfg.temperature, cfg.top_k, cfg.top_p)
            chosen = torch.multinomial(torch.softmax(filtered, dim=-1), 1, generator=generator).squeeze(-1)
        for row in range(bsz):
            if done[row]:
                continue
            token = int(chosen[row])
            if token in stop:
                done[row] = True
                status[row] = stop[token]
            else:
                outputs[row].append(token)
        if all(done):
            break
        feed = chosen.clone()
        for row in range(bsz):
            if done[row]:
                feed[row] = tok.pad_id
        mask = torch.cat([mask, torch.ones((bsz, 1), dtype=mask.dtype, device=device)], dim=1)
        logits, past = model(feed.unsqueeze(1).to(device), attention_mask=mask, past_kv=past, use_cache=True)
        next_logits = logits[:, -1, :]

    elapsed = time.time() - started
    results = []
    for row in range(bsz):
        text = tok.decode(outputs[row]).strip()
        st = status[row]
        if st == STATUS_OK and not text:
            st = STATUS_EMPTY
        results.append(Generation(list(prompts[row]), outputs[row], st, text, elapsed / bsz))
    return results

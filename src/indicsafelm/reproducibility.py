"""Standard-library reproducibility helpers used by the Phase 0 smoke path."""

import os
import random


def seed_python(seed: int, set_hash_seed: bool = True) -> None:
    """Seed Python's RNG and optionally set the hash seed for child processes."""

    random.seed(seed)
    if set_hash_seed:
        os.environ.setdefault("PYTHONHASHSEED", str(seed))


def stable_sample(seed: int, size: int = 8) -> list:
    """Return a deterministic sample useful for smoke-test comparisons."""

    if size < 0:
        raise ValueError("size must be non-negative")
    generator = random.Random(seed)
    return [generator.random() for _ in range(size)]


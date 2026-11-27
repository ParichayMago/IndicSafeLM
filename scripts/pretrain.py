"""Entry point: pretrain the base checkpoint (see indicsafelm.training.pretrain)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indicsafelm.training.pretrain import PretrainConfig, run_pretraining  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_base.json")
    parser.add_argument("--resume", type=Path, help="run directory's resume/ folder")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--minutes", type=float)
    parser.add_argument("--export-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    cfg = PretrainConfig(**json.loads(args.config.read_text())) if args.config.exists() else PretrainConfig()
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.minutes is not None:
        cfg.time_budget_minutes = args.minutes
    if args.export_dir:
        cfg.export_dir = args.export_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    manifest = run_pretraining(cfg, ROOT, resume=args.resume, log=lambda m: print(m, flush=True))
    print(json.dumps({k: manifest[k] for k in ("run_id", "completed_steps", "wall_minutes", "best_checkpoint")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

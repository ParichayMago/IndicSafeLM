"""Run the frozen held-out evaluation on every available checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indicsafelm.evaluation.run import markdown_report, run_evaluation  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="evaluate only the first N prompts (smoke test)")
    args = parser.parse_args()
    result = run_evaluation(ROOT, limit=args.limit, log=lambda m: print(m, flush=True))
    print(markdown_report(result))

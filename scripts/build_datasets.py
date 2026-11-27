"""Build SFT, preference, PPO-prompt, dev/test evaluation and demo datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indicsafelm.instructions.build import build_all  # noqa: E402

if __name__ == "__main__":
    manifest = build_all(ROOT)
    print(json.dumps({k: v["rows"] for k, v in manifest["files"].items()}, indent=2))
    print("SFT response behaviours:", manifest["sft_response_behavior"])
    print("preference rejected types:", manifest["pref_rejected_reason"])

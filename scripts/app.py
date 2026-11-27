"""Launch IndicSafe Sahayak, the student messaging assistant: python scripts/app.py [--port 7862]."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indicsafelm.demo.sahayak_app import main  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7862)
    main(port=parser.parse_args().port)

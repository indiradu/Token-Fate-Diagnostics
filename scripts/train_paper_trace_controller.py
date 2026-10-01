from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regret_remasking.paper_trace import main


if __name__ == "__main__":
    raise SystemExit(main())

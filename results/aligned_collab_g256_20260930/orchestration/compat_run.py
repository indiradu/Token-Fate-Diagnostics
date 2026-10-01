"""Run one of the collaborator's scripts unchanged, after installing collab_compat."""
from __future__ import annotations

import runpy
import sys

import collab_compat  # noqa: F401  (patches her load_llada before the script imports it)

if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = [script] + sys.argv[2:]
    runpy.run_path(script, run_name="__main__")

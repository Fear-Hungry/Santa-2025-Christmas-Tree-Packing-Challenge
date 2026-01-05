#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for cand in (here.parent, *here.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    return here.parent


def main() -> int:
    root = _repo_root()
    cmd = [sys.executable, "-m", "santa_packing.cli.generate_submission", *sys.argv[1:]]
    return int(subprocess.call(cmd, cwd=str(root)))


if __name__ == "__main__":
    raise SystemExit(main())

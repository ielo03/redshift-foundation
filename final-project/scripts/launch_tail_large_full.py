from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "final_project_venv" / "Scripts" / "python.exe"
OUTPUT_DIR = ROOT / "outputs" / "redshift_tail_large_full"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON),
        "-u",
        str(ROOT / "scripts" / "train_redshift_tail_scaling.py"),
        "--model-size",
        "large",
        "--input",
        str(ROOT / "data" / "processed" / "desi_sv3_primary_full_clean.pt"),
        "--output-dir",
        str(OUTPUT_DIR),
        "--epochs",
        "60",
        "--batch-size",
        "8",
        "--gradient-accumulation",
        "8",
        "--disable-early-stopping",
    ]
    stdout = open(OUTPUT_DIR / "train_stdout.log", "ab", buffering=0)
    stderr = open(OUTPUT_DIR / "train_stderr.log", "ab", buffering=0)
    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=stdout,
        stderr=stderr,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    (OUTPUT_DIR / "train_pid.txt").write_text(f"{process.pid}\n", encoding="ascii")
    print(process.pid)


if __name__ == "__main__":
    if not PYTHON.exists():
        sys.exit(f"Missing venv interpreter: {PYTHON}")
    main()

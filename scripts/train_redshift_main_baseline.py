from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).with_name("train_redshift_baseline.py")
    baseline_output_dir = "hw/final_project/outputs/redshift_baseline"
    prefixed_args = [
        str(script_path),
        "--output-dir",
        baseline_output_dir,
        "--alpha",
        "1.0",
        "--redshift-loss",
        "mse",
        "--sampling-strategy",
        "uniform",
        "--tail-power",
        "1.0",
        "--mask-strategy",
        "random",
        "--mask-prob",
        "0.15",
        "--mask-seed",
        "42",
    ]
    sys.argv = prefixed_args + sys.argv[1:]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()

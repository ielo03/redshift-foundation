from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).with_name("train_redshift_baseline.py")
    default_args = [
        str(script_path),
        "--input",
        "data/processed/mmu_desi_hf_edr_sv3.pt",
        "--output-dir",
        "outputs/redshift_tail_best_small_hf_100k",
        "--d-model",
        "256",
        "--num-layers",
        "4",
        "--nhead",
        "8",
        "--patch-size",
        "61",
        "--alpha",
        "1.25",
        "--reconstruction-weight",
        "1.0",
        "--redshift-loss",
        "smoothl1",
        "--sampling-strategy",
        "tail",
        "--tail-power",
        "1.75",
        "--mask-prob",
        "0.15",
        "--best-metric",
        "val_mae",
    ]
    sys.argv = default_args + sys.argv[1:]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()

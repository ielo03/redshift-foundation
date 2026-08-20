from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).with_name("train_redshift_baseline.py")
    variant_output_dir = "hw/final_project/outputs/redshift_uncertainty_variant"
    prefixed_args = [
        str(script_path),
        "--output-dir",
        variant_output_dir,
        "--alpha",
        "1.25",
        "--redshift-loss",
        "smoothl1",
        "--redshift-objective",
        "relative",
        "--sampling-strategy",
        "tail",
        "--tail-power",
        "1.75",
        "--use-ivar-channel",
        "--use-validity-channel",
        "--stage2-epochs",
        "10",
        "--stage2-alpha",
        "2.5",
        "--stage2-mask-prob",
        "0.05",
        "--stage2-lr",
        "3e-4",
    ]
    sys.argv = prefixed_args + sys.argv[1:]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()

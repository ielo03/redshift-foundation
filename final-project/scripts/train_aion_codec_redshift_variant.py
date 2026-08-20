from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).with_name("train_aion_codec_baseline.py")
    variant_output_dir = "hw/final_project/outputs/aion_codec_redshift_variant"
    prefixed_args = [
        str(script_path),
        "--output-dir",
        variant_output_dir,
        "--alpha",
        "3.0",
        "--mask-seed",
        "84",
    ]
    sys.argv = prefixed_args + sys.argv[1:]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()

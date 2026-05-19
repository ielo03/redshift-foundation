from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).with_name("train_redshift_baseline.py")
    foundation_args = [
        str(script_path),
        "--architecture",
        "redshift_token",
        "--redshift-loss",
        "smoothl1",
        "--sampling-strategy",
        "tail",
        "--tail-power",
        "1.75",
        "--best-metric",
        "val_mae",
    ]
    sys.argv = foundation_args + sys.argv[1:]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()

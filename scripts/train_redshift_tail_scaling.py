from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


MODEL_SIZE_ARGS = {
    "small": ["--d-model", "256", "--num-layers", "4", "--nhead", "8", "--patch-size", "61"],
    "medium": ["--d-model", "384", "--num-layers", "6", "--nhead", "8", "--patch-size", "61"],
    "large": ["--d-model", "512", "--num-layers", "8", "--nhead", "8", "--patch-size", "61"],
}


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Scale the best-performing tail-focused raw-spectrum redshift variant.",
        add_help=False,
    )
    parser.add_argument("--model-size", choices=sorted(MODEL_SIZE_ARGS), default="small")
    return parser.parse_known_args(argv)


def main() -> None:
    wrapper_args, passthrough_args = parse_wrapper_args(sys.argv[1:])
    script_path = Path(__file__).with_name("train_redshift_baseline.py")
    default_output_dir = f"hw/final_project/outputs/redshift_tail_{wrapper_args.model_size}_scaling"
    best_variant_args = [
        str(script_path),
        "--output-dir",
        default_output_dir,
        "--alpha",
        "1.25",
        "--redshift-loss",
        "smoothl1",
        "--sampling-strategy",
        "tail",
        "--tail-power",
        "1.75",
        *MODEL_SIZE_ARGS[wrapper_args.model_size],
    ]
    sys.argv = best_variant_args + passthrough_args
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()

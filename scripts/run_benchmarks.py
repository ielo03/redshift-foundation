#!/usr/bin/env python3
"""Run the fixed DESI benchmark and SDSS OOD benchmark for one checkpoint."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sdss-bundle", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("data/preprocessed/split_manifest_v1.jsonl"))
    p.add_argument("--max-spectra", type=int, default=10_000)
    p.add_argument("--examples", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    # Accepted only so already-submitted Slurm scripts remain runnable. Full
    # spectrum benchmarks intentionally ignore reconstruction masking.
    p.add_argument("--mask-prob", type=float, help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    return p.parse_args()


def run(command: list[str]) -> None:
    print("[benchmark]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    a = parse_args()
    if not a.model.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {a.model}")
    if not a.sdss_bundle.is_file():
        raise FileNotFoundError(f"Missing prepared SDSS bundle: {a.sdss_bundle}")
    common = ["--model", str(a.model), "--max-spectra", str(a.max_spectra), "--examples", str(a.examples), "--batch-size", str(a.batch_size)]
    root = Path(__file__).resolve().parent
    run([sys.executable, str(root / "evaluate_desi_benchmark.py"), "--manifest", str(a.manifest), "--output-dir", str(a.output_dir / "desi_benchmark"), *common])
    run([sys.executable, str(root / "evaluate_ood_sdss.py"), "--input", str(a.sdss_bundle), "--output-dir", str(a.output_dir / "sdss_ood"), *common])


if __name__ == "__main__":
    main()

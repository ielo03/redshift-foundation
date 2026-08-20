from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine processed spectra bundles into one lean trainer bundle.")
    parser.add_argument("--input", nargs="+", type=Path, required=True, help="Processed .pt bundles to combine.")
    parser.add_argument("--output", type=Path, required=True, help="Output combined .pt bundle.")
    parser.add_argument("--summary", type=Path, required=True, help="Output summary JSON.")
    parser.add_argument(
        "--lean",
        action="store_true",
        help="Keep only flux, z, and one wavelength grid to reduce RAM usage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flux_chunks = []
    z_chunks = []
    wavelength = None
    sources = []

    for path in args.input:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        batch = bundle.get("batch", bundle)
        flux_chunks.append(batch["flux"].contiguous())
        z_chunks.append(batch["z"].contiguous())

        if wavelength is None:
            w = batch["wavelength"]
            wavelength = w[0].contiguous() if w.ndim == 2 else w.contiguous()

        sources.append(str(path))
        del bundle, batch

    flux = torch.cat(flux_chunks, dim=0)
    z = torch.cat(z_chunks, dim=0)
    batch_out = {"flux": flux, "z": z, "wavelength": wavelength}

    summary = {
        "count": int(z.shape[0]),
        "length_min": int(flux.shape[1]),
        "length_max": int(flux.shape[1]),
        "length_mean": float(flux.shape[1]),
        "z_min": float(z.min().item()),
        "z_max": float(z.max().item()),
        "z_mean": float(z.float().mean().item()),
        "lean": bool(args.lean),
        "sources": sources,
    }

    payload = {"source": sources, "summary": summary, "batch": batch_out}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved combined bundle:", args.output)
    print("Saved summary:", args.summary)
    print("Combined spectra:", summary["count"])


if __name__ == "__main__":
    main()

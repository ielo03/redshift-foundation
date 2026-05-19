from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_src_to_path(project_root: Path) -> None:
    src_dir = project_root / "src"
    sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a predicted-vs-actual redshift scatter plot."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional JSON file with keys 'actual' and 'predicted'. If omitted, uses synthetic values.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to outputs/figures/redshift_scatter.png.",
    )
    return parser.parse_args()


def load_pairs(input_path: Path | None) -> tuple[list[float], list[float]]:
    if input_path is None:
        actual = [0.02, 0.10, 0.21, 0.35, 0.48]
        predicted = [0.03, 0.09, 0.19, 0.40, 0.44]
        return actual, predicted

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    actual = payload.get("actual")
    predicted = payload.get("predicted")
    if actual is None or predicted is None:
        raise KeyError("input JSON must contain 'actual' and 'predicted' keys")
    return actual, predicted


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    add_src_to_path(project_root)

    from redshift_diagnostics import plot_predicted_vs_actual_redshift, redshift_metrics

    actual, predicted = load_pairs(args.input)
    output_path = args.output or (project_root / "outputs" / "figures" / "redshift_scatter.png")

    metrics = redshift_metrics(actual, predicted)
    saved_path = plot_predicted_vs_actual_redshift(actual, predicted, output_path)

    print("Metrics:", metrics)
    print("Saved plot to:", saved_path)


if __name__ == "__main__":
    main()

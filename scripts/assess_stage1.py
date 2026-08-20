#!/usr/bin/env python3
"""Decide whether a stage-1 experiment is clearly bad enough to stop."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


PRIMARY_METRICS = ("z_mae", "z_sigma_nmad")


def load_history(path: Path, through_epoch: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if int(row["epoch"]) <= through_epoch]
    if not rows or int(rows[-1]["epoch"]) != through_epoch:
        raise RuntimeError(f"{path} does not contain a completed epoch {through_epoch}")
    return rows


def best_value(rows: list[dict], metric: str) -> float:
    return min(float(row["val"][metric]) for row in rows)


def recent_improvement(rows: list[dict], metric: str, window: int) -> float:
    if len(rows) < 2 * window:
        raise RuntimeError(f"Need at least {2 * window} epochs for two trend windows")
    previous = statistics.median(float(row["val"][metric]) for row in rows[-2 * window : -window])
    recent = statistics.median(float(row["val"][metric]) for row in rows[-window:])
    return (previous - recent) / max(abs(previous), 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-history", type=Path, required=True)
    parser.add_argument("--control-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-epoch", type=int, default=64)
    parser.add_argument("--worse-factor", type=float, default=1.5)
    parser.add_argument("--trend-window", type=int, default=12)
    parser.add_argument("--minimum-improving-trend", type=float, default=0.05)
    args = parser.parse_args()

    candidate = load_history(args.candidate_history, args.stage_epoch)
    control = load_history(args.control_history, args.stage_epoch)
    candidate_best = {metric: best_value(candidate, metric) for metric in PRIMARY_METRICS}
    control_best = {metric: best_value(control, metric) for metric in PRIMARY_METRICS}
    ratios = {metric: candidate_best[metric] / control_best[metric] for metric in PRIMARY_METRICS}
    trends = {
        metric: recent_improvement(candidate, metric, args.trend_window)
        for metric in PRIMARY_METRICS
    }
    finite = all(
        math.isfinite(value)
        for values in (candidate_best, control_best, ratios, trends)
        for value in values.values()
    )
    much_worse_on_both = all(ratios[metric] >= args.worse_factor for metric in PRIMARY_METRICS)
    improving = any(trends[metric] >= args.minimum_improving_trend for metric in PRIMARY_METRICS)
    stop = (not finite) or (much_worse_on_both and not improving)
    reasons = []
    if not finite:
        reasons.append("non-finite primary validation metric")
    if much_worse_on_both:
        reasons.append(f"both primary metrics are at least {args.worse_factor:.2f}x Control")
    if improving:
        reasons.append(
            f"at least one primary metric improved by {args.minimum_improving_trend:.0%} in the recent trend windows"
        )
    decision = {
        "decision": "stop" if stop else "continue",
        "stage_epoch": args.stage_epoch,
        "candidate_best": candidate_best,
        "control_best": control_best,
        "candidate_to_control_ratios": ratios,
        "recent_fractional_improvement": trends,
        "worse_factor": args.worse_factor,
        "minimum_improving_trend": args.minimum_improving_trend,
        "reasons": reasons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2), flush=True)
    return 2 if stop else 0


if __name__ == "__main__":
    raise SystemExit(main())

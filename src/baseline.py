from __future__ import annotations

import json
from datetime import datetime

from project_paths import FIGURES_DIR, TABLES_DIR, ensure_project_dirs


def main() -> None:
    ensure_project_dirs()

    summary = {
        "status": "starter baseline ran successfully",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "next_steps": [
            "choose a dataset",
            "implement preprocessing",
            "replace this file with a real baseline model",
        ],
    }

    output_path = TABLES_DIR / "baseline_summary.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Created:", output_path)
    print("Figures directory ready at:", FIGURES_DIR)


if __name__ == "__main__":
    main()

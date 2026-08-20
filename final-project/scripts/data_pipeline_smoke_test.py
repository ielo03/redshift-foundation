from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_path(project_root: Path) -> None:
    src_dir = project_root / "src"
    sys.path.insert(0, str(src_dir))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    add_src_to_path(project_root)

    from data_pipeline import collate_records, summarize_records, synthetic_records

    records = synthetic_records(batch_size=3, length=128)
    summary = summarize_records(records)
    batch = collate_records(records, target_length=128)

    print("Synthetic record summary:")
    print(summary)
    print("Batch tensor shapes:")
    for key, value in batch.items():
        print(f"- {key}: shape={tuple(value.shape)} dtype={value.dtype}")


if __name__ == "__main__":
    main()


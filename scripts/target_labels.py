"""DESI primary-target labels used for multi-label training and evaluation.

These are target-selection labels, not mutually exclusive spectral classes:
BGS is selected from BGS_TARGET, while LRG/ELG/QSO are the corresponding
primary bits in the survey-specific DESI_TARGET mask.  The bit positions are
stable across the DR1 main and survey-validation targeting schemas.
"""
from __future__ import annotations

import numpy as np


TARGET_LABEL_NAMES = ("BGS", "LRG", "ELG", "QSO")
_DESI_BITS = {"LRG": 0, "ELG": 1, "QSO": 2}


def target_column_names(survey: str, names: tuple[str, ...] | list[str]) -> tuple[str, str]:
    available = set(names)
    prefix = survey.upper() if survey.lower().startswith("sv") else ""
    desi = f"{prefix + '_' if prefix else ''}DESI_TARGET"
    bgs = f"{prefix + '_' if prefix else ''}BGS_TARGET"
    # Some cumulative DR1 fibermaps retain both SV and main columns.  Always
    # prefer the schema matching the record's survey, then use the main column.
    if desi not in available:
        desi = "DESI_TARGET"
    if bgs not in available:
        bgs = "BGS_TARGET"
    if desi not in available or bgs not in available:
        raise KeyError(f"Missing target bitmask columns for survey={survey}: need {desi} and {bgs}")
    return desi, bgs


def target_selection_labels(fibermap: np.ndarray, survey: str) -> np.ndarray:
    """Return Bx4 uint8 [BGS, LRG, ELG, QSO] target-selection labels."""

    if fibermap.dtype.names is None:
        raise TypeError("Expected a structured FIBERMAP array")
    desi_column, bgs_column = target_column_names(survey, fibermap.dtype.names)
    desi = np.asarray(fibermap[desi_column], dtype=np.uint64)
    bgs = np.asarray(fibermap[bgs_column], dtype=np.uint64)
    return np.stack(
        [
            bgs != 0,
            (desi & (np.uint64(1) << np.uint64(_DESI_BITS["LRG"]))) != 0,
            (desi & (np.uint64(1) << np.uint64(_DESI_BITS["ELG"]))) != 0,
            (desi & (np.uint64(1) << np.uint64(_DESI_BITS["QSO"]))) != 0,
        ],
        axis=1,
    ).astype(np.uint8)

from __future__ import annotations
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np

from .config import CLASS_MAP, DATA_DIR, SEGMENT_SAMPLES
from .preprocess import read_record, read_annotations, make_beat_segments


def build_dataset(
    data_dir: Path | str | None = None,
    class_map: Dict[str, int] | None = None,
    records: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build arrays X (num_beats, SEGMENT_SAMPLES) and y (num_beats,) from local MIT-BIH files.

    - Uses MLII/II channel when available
    - Segments 1000-sample windows centered on R, z-score per segment
    - Keeps only beats whose annotation symbol is in class_map

    Returns:
        X: float32 array (N, SEGMENT_SAMPLES)
        y: int64 array (N,)
        rec_ids: list of record ids for each beat
    """
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    cmap = class_map if class_map is not None else CLASS_MAP

    if records is None:
        # Infer record names from .hea files
        records = sorted([p.stem for p in base.glob("*.hea")])

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    rec_ids: List[str] = []

    for rec in records:
        rec_path = base / rec
        if not (rec_path.with_suffix(".hea").exists() and rec_path.with_suffix(".dat").exists()):
            continue
        try:
            sig, _ = read_record(rec_path)
            r_locs, symbols = read_annotations(rec_path)
        except Exception:
            continue
        beats = make_beat_segments(rec, sig, r_locs, symbols)
        for b in beats:
            if b.symbol in cmap:
                X_list.append(b.segment)
                y_list.append(cmap[b.symbol])
                rec_ids.append(rec)

    if not X_list:
        raise RuntimeError(f"No beats found in {base}. Ensure dataset is downloaded and contains annotations.")

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, rec_ids

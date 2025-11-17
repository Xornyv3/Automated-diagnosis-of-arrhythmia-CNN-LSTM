from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import wfdb

from .config import FS, PRE_SAMPLES, POST_SAMPLES, SEGMENT_SAMPLES


@dataclass
class BeatSegment:
    record: str
    r_index: int
    segment: np.ndarray  # shape (SEGMENT_SAMPLES,)
    symbol: str  # MIT-BIH annotation symbol


def read_record(record_path: str | Path) -> Tuple[np.ndarray, List[str]]:
    """
    Read a WFDB record and return the signal (prefer MLII) and channel names.
    """
    rec = wfdb.rdrecord(str(record_path))
    sig = rec.p_signal  # float64 array shape (n_samples, n_channels)
    ch_names = [str(n) for n in rec.sig_name]

    # Prefer MLII or II; fallback to the first channel
    channel_idx = None
    for preferred in ("MLII", "II"):
        if preferred in ch_names:
            channel_idx = ch_names.index(preferred)
            break
    if channel_idx is None:
        channel_idx = 0
    lead = sig[:, channel_idx]
    return lead.astype(np.float32), ch_names


def read_annotations(record_path: str | Path) -> Tuple[np.ndarray, List[str]]:
    """
    Read 'atr' annotations to obtain R peak sample indices and symbols.
    """
    ann = wfdb.rdann(str(record_path), "atr")
    return np.asarray(ann.sample, dtype=np.int64), [str(s) for s in ann.symbol]


def segment_around_r(signal: np.ndarray, r_index: int) -> np.ndarray:
    """Extract a fixed-length segment around an R-peak with zero padding as needed."""
    start = r_index - PRE_SAMPLES
    end = r_index + POST_SAMPLES
    seg = np.zeros((SEGMENT_SAMPLES,), dtype=np.float32)

    src_start = max(start, 0)
    src_end = min(end, signal.shape[0])
    dst_start = src_start - start  # if start<0, we shift into the destination
    dst_end = dst_start + (src_end - src_start)
    if src_end > src_start:
        seg[dst_start:dst_end] = signal[src_start:src_end]
    return seg


def zscore_per_segment(seg: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = float(np.mean(seg))
    s = float(np.std(seg))
    return (seg - m) / (s + eps)


def make_beat_segments(
    record: str,
    signal: np.ndarray,
    r_locs: np.ndarray,
    symbols: List[str],
) -> List[BeatSegment]:
    beats: List[BeatSegment] = []
    for r, sym in zip(r_locs, symbols):
        seg = segment_around_r(signal, int(r))
        seg = zscore_per_segment(seg)
        beats.append(BeatSegment(record=record, r_index=int(r), segment=seg, symbol=sym))
    return beats

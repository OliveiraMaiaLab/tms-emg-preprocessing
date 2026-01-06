"""
utils/mep_loading.py
--------------------
Extract per-block MEP waveforms from raw EMG using pulse indices stored in session JSON.

Returns:
  - t_ms: (T,) time axis in milliseconds (relative to pulse)
  - meps: (N, T) waveforms (baseline-corrected, in uV)
"""

from __future__ import annotations

from typing import Tuple, List
from pathlib import Path
import numpy as np
import json

from utils.tms_module import load_data


def load_meps_for_block(
    meta: dict,
    session_file: str | Path,
    block_name: str,
    hemi: str = "left",
    pre_s: float = 0.100,
    post_s: float = 0.400,
) -> Tuple[np.ndarray, np.ndarray]:
    session_file = Path(session_file)

    js = json.loads(session_file.read_text())
    meps_root = js.get("meps", {}) or {}
    hp = ((meps_root.get(block_name) or {}).get(hemi) or {})
    pulses = hp.get("pulses", [])
    if not isinstance(pulses, list) or not pulses:
        return np.asarray([], dtype=float), np.zeros((0, 0), dtype=float)

    fs = float(meta["sampling_rate"])
    data, _tms = load_data(meta["input_file"], channels=meta.get("channels"))

    hemi_to_ch = {"left": 0, "right": 1}
    ch = hemi_to_ch.get(hemi, 0)
    sig = data[ch, :]

    pre_n = int(round(pre_s * fs))
    post_n = int(round(post_s * fs))
    n_win = pre_n + post_n

    t_ms = (np.arange(n_win) - pre_n) / fs * 1000.0

    trials = []
    for p in pulses:
        p = int(p)
        a = p - pre_n
        b = p + post_n
        if a < 0 or b > sig.shape[0]:
            continue
        y = sig[a:b].astype(float)
        y -= float(y[:pre_n].mean())  # baseline correct
        trials.append(y)

    if not trials:
        return t_ms, np.zeros((0, n_win), dtype=float)

    meps = np.asarray(trials, dtype=float)
    return t_ms, meps

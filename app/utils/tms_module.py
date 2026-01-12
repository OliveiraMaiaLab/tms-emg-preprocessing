# app/utils/tms_module.py
# -*- coding: utf-8 -*-
"""
tms_module.py
-------------
Core signal + session helpers used throughout the Streamlit GUI.

Keep this module lean: it should contain only functions that are used by the app.

Key responsibilities:
- Load binary EMG (.bin)
- Extract per-block MEP epochs (using pulses stored in session JSON)
- Minimal JSON read/write helpers
- Epoch handling (mep_window -> ms)
- peaks_flag helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import json
import numpy as np


# =============================================================================
# Session / JSON helpers
# =============================================================================
@dataclass(frozen=True)
class Epoch:
    """Epoch bounds in milliseconds (relative to pulse)."""
    tmin_ms: float
    tmax_ms: float


def read_json(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def get_epoch_from_session(session: dict) -> Epoch:
    """
    Reads epoch bounds from:
      session["mep_window"] = [beg_s, end_s]   (seconds, relative to pulse)
    Returns in milliseconds.
    """
    w = session.get("mep_window", None)
    if not (isinstance(w, list) and len(w) == 2 and w[0] is not None and w[1] is not None):
        raise KeyError(
            "Missing MEP window definition in session JSON. "
            "Expected session['mep_window'] = [beg_s, end_s]."
        )
    beg_s = float(w[0])
    end_s = float(w[1])
    if end_s <= beg_s:
        raise ValueError(f"Invalid mep_window: beg_s={beg_s} must be < end_s={end_s}")
    return Epoch(beg_s * 1000.0, end_s * 1000.0)


def get_peaks_flag_list(session: dict, block: str, hemi: str = "left") -> List[int]:
    meps = session.get("meps", {}) or {}
    hp = ((meps.get(str(block)) or {}).get(str(hemi)) or {})
    flags = hp.get("peaks_flag", [])
    if not isinstance(flags, list):
        return []
    out: list[int] = []
    for v in flags:
        try:
            out.append(1 if int(v) else 0)
        except Exception:
            out.append(0)
    return out


def set_peaks_flag_list(session: dict, block: str, flags: List[int], hemi: str = "left") -> None:
    session.setdefault("meps", {})
    session["meps"].setdefault(str(block), {})
    session["meps"][str(block)].setdefault(str(hemi), {})
    hp = session["meps"][str(block)][str(hemi)]
    hp["peaks_flag"] = [1 if int(v) else 0 for v in flags]
    session["meps"][str(block)][str(hemi)] = hp


def save_peaks_flag_list(session_file: str | Path, block: str, flags: List[int], hemi: str = "left") -> None:
    session_file = Path(session_file)
    session = read_json(session_file)
    set_peaks_flag_list(session, block, flags, hemi=hemi)
    write_json(session_file, session)


# =============================================================================
# Binary EMG loading
# =============================================================================
def load_data(
    file_path: str | Path,
    *,
    channels: np.ndarray = np.array((3, 1, 2)),
    fs: int = 4000,
    normalize_voltage: bool = True,
    voltage_gain: float = 133.0,
    reshape: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load EMG from a uint16 interleaved binary file and return:
      data: shape (2, n_samples) as [left, right] in uV (if normalize_voltage)
      tms_indexes: rising edges from synch channel

    `channels` is expected as: (synch_pulse, right, left)
    (this matches your template convention and previous code).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"EMG file not found: {file_path}")

    c_synch = int(channels[0])
    c_right = int(channels[1])
    c_left = int(channels[2])

    raw = np.fromfile(str(file_path), dtype="uint16").astype(float)
    if raw.size % reshape != 0:
        raise ValueError(f"File length not divisible by reshape={reshape}: {file_path}")

    raw = raw.reshape((reshape, raw.size // reshape), order="F")
    n_samples = raw.shape[1]
    idx = np.arange(n_samples)

    if normalize_voltage:
        # convert ADC to volts (3V full-scale, 16-bit), center at 1.5V, divide by gain, -> uV
        left = ((raw[c_left, idx] * (3.0 / (2**16)) - 1.5) / float(voltage_gain)) * 1e6
        right = ((raw[c_right, idx] * (3.0 / (2**16)) - 1.5) / float(voltage_gain)) * 1e6
    else:
        left = raw[c_left, idx]
        right = raw[c_right, idx]

    data = np.asarray([left, right], dtype=float)

    pulse_signal = raw[c_synch, idx]
    threshold = pulse_signal.mean() + 3.0 * pulse_signal.std()
    above = pulse_signal > threshold
    tms_indexes = np.where(np.diff(above.astype(int)) == 1)[0] + 1  # rising edges

    return data, tms_indexes.astype(int)


# =============================================================================
# MEP extraction (per-block)
# =============================================================================
def load_meps_for_block(
    meta: dict,
    session_file: str | Path,
    block_name: str,
    *,
    hemi: str = "left",
    detrend: bool = True,
    epoch: Epoch | None = None,  # if provided, use epoch bounds
    pre_s: float = 0.100,        # used only if epoch is None
    post_s: float = 0.400,       # used only if epoch is None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract per-block MEP waveforms from raw EMG using pulse indices stored in session JSON.

    Returns:
      t_ms: (T,) time axis in ms relative to pulse
      meps: (N, T) waveforms (uV)
    """
    session_file = Path(session_file)
    js = read_json(session_file)

    meps_root = js.get("meps", {}) or {}
    hp = ((meps_root.get(str(block_name)) or {}).get(str(hemi)) or {})
    pulses = hp.get("pulses", [])
    if not isinstance(pulses, list) or len(pulses) == 0:
        return np.asarray([], dtype=float), np.zeros((0, 0), dtype=float)

    fs = float(meta.get("sampling_rate", 4000))

    # meta["input_file"] may be a full path or just a filename (step_input can resolve);
    # if it's a filename, prefer joining with meta["data_dir"].
    input_file = meta.get("input_file", "")
    data_dir = meta.get("data_dir", "")

    in_path = Path(str(input_file))
    if not in_path.exists() and data_dir:
        candidate = Path(str(data_dir)) / str(input_file)
        if candidate.exists():
            in_path = candidate

    data, _tms = load_data(in_path, channels=np.asarray(meta.get("channels")))

    hemi_to_ch = {"left": 0, "right": 1}
    ch = hemi_to_ch.get(str(hemi).lower(), 0)
    sig = data[ch, :]

    if epoch is not None:
        start_ms = float(epoch.tmin_ms)
        end_ms = float(epoch.tmax_ms)
    else:
        start_ms = -float(pre_s) * 1000.0
        end_ms = float(post_s) * 1000.0

    if end_ms <= start_ms:
        raise ValueError(f"Invalid window: start_ms={start_ms} must be < end_ms={end_ms}")

    start_n = int(np.round(start_ms / 1000.0 * fs))
    end_n = int(np.round(end_ms / 1000.0 * fs))
    n_win = end_n - start_n
    if n_win <= 0:
        raise ValueError("Window length is zero/negative after sample conversion.")

    t_ms = (np.arange(start_n, end_n) / fs) * 1000.0

    trials: list[np.ndarray] = []
    for p in pulses:
        p = int(p)
        a = p + start_n
        b = p + end_n
        if a < 0 or b > sig.shape[0]:
            continue
        y = sig[a:b].astype(float)
        if detrend:
            y -= float(y.mean())
        trials.append(y)

    if not trials:
        return t_ms, np.zeros((0, n_win), dtype=float)

    meps = np.asarray(trials, dtype=float)
    return t_ms, meps

# app/utils/tms_module.py
# -*- coding: utf-8 -*-
"""
tms_module.py
-------------
Core signal + session helpers used throughout the Streamlit GUI.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import json
import numpy as np

from scipy.signal import detrend as sp_detrend, savgol_filter
from skimage.restoration import denoise_wavelet


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
    tmp  = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def get_epoch_from_session(session: dict) -> Epoch:
    w = session.get("mep_window", None)
    if not (isinstance(w, list) and len(w) == 2 and w[0] is not None and w[1] is not None):
        raise KeyError(
            "Missing MEP window definition in session JSON. "
            "Expected session['mep_window'] = [beg_s, end_s]."
        )
    beg_s, end_s = float(w[0]), float(w[1])
    if end_s <= beg_s:
        raise ValueError(f"Invalid mep_window: beg_s={beg_s} must be < end_s={end_s}")
    return Epoch(beg_s * 1000.0, end_s * 1000.0)


# =============================================================================
# Generic per-MEP flag helpers
# =============================================================================
def get_flag_list(session: dict, block: str, hemi: str, flag_name: str) -> List[int]:
    """
    Read a named flag list from session["meps"][block][hemi][flag_name].
    Returns a list of ints (0/1); missing/malformed values become 0.
    """
    meps = session.get("meps", {}) or {}
    hp   = ((meps.get(str(block)) or {}).get(str(hemi)) or {})
    flags = hp.get(flag_name, [])
    if not isinstance(flags, list):
        return []
    out: list[int] = []
    for v in flags:
        try:
            out.append(1 if int(v) else 0)
        except Exception:
            out.append(0)
    return out


def save_flag_list(
    session_file: str | Path,
    block: str,
    flags: List[int],
    hemi: str,
    flag_name: str,
) -> None:
    """Write a named flag list back to the session JSON (atomic)."""
    session_file = Path(session_file)
    session = read_json(session_file)
    session.setdefault("meps", {})
    session["meps"].setdefault(str(block), {})
    session["meps"][str(block)].setdefault(str(hemi), {})
    hp = session["meps"][str(block)][str(hemi)]
    hp[flag_name] = [1 if int(v) else 0 for v in flags]
    session["meps"][str(block)][str(hemi)] = hp
    write_json(session_file, session)


# ---------------------------------------------------------------------------
# Convenience wrappers kept for backward compatibility
# ---------------------------------------------------------------------------
def get_peaks_flag_list(session: dict, block: str, hemi: str = "left") -> List[int]:
    return get_flag_list(session, block, hemi, "peaks_flag")


def set_peaks_flag_list(session: dict, block: str, flags: List[int], hemi: str = "left") -> None:
    session.setdefault("meps", {})
    session["meps"].setdefault(str(block), {})
    session["meps"][str(block)].setdefault(str(hemi), {})
    hp = session["meps"][str(block)][str(hemi)]
    hp["peaks_flag"] = [1 if int(v) else 0 for v in flags]
    session["meps"][str(block)][str(hemi)] = hp


def save_peaks_flag_list(
    session_file: str | Path, block: str, flags: List[int], hemi: str = "left"
) -> None:
    save_flag_list(session_file, block, flags, hemi, "peaks_flag")


# =============================================================================
# Signal denoising / filtering
# =============================================================================
def denoise_signal(
    sig: np.ndarray,
    *,
    sigma: float | None = None,
    wavelet: str = "db1",
    mode: str = "soft",
    wavelet_levels: int = 3,
    method: str = "BayesShrink",
    rescale_sigma: bool = True,
    sg_window: int = 5,
    sg_order: int = 3,
) -> np.ndarray:
    """
    Denoise one continuous EMG channel.

    Pipeline (applied in order):
      1. Wavelet denoising — `wavelet_levels`-level Daubechies (`wavelet`) with
         `method` (BayesShrink) soft thresholding.
      2. Savitzky-Golay smoothing (window `sg_window` samples, order `sg_order`),
         preferred over band-pass filtering to limit waveform distortion.
      3. Linear detrend to remove slow baseline drift.

    `sigma` is the noise standard deviation (in the same units as `sig`, i.e. uV).
    Leave it None (default) to let skimage estimate it per channel from the
    finest wavelet detail coefficients (robust MAD estimate). Because MEPs are
    sparse (~0.2% of samples in a typical recording), this estimate reads the
    baseline noise floor reliably and is not inflated by the events. Pass a fixed
    constant only if you want to hold the denoising strength constant across
    channels/recordings instead.

    Returns the filtered channel (same shape as `sig`, in uV).
    """
    sig = np.asarray(sig, dtype=float)

    den = denoise_wavelet(
        sig,
        wavelet=wavelet,
        mode=mode,
        wavelet_levels=wavelet_levels,
        method=method,
        rescale_sigma=rescale_sigma,
        sigma=sigma,
    )
    smoothed = savgol_filter(den, sg_window, sg_order)
    return sp_detrend(smoothed)


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
    filter_signal: bool = True,
    sigma: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load EMG from a uint16 interleaved binary file.
    When `filter_signal` is True, each EMG channel is denoised on load
    (wavelet + Savitzky-Golay + detrend); the synch channel is left raw so
    pulse detection is unaffected. `sigma` is the fixed noise std (uV) passed
    to the wavelet denoiser; None lets skimage estimate it per channel.
    Returns:
      data: (2, n_samples) [left, right] in uV
      tms_indexes: rising edges from synch channel
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"EMG file not found: {file_path}")

    c_synch = int(channels[0])
    c_right = int(channels[1])
    c_left  = int(channels[2])

    raw = np.fromfile(str(file_path), dtype="uint16").astype(float)
    if raw.size % reshape != 0:
        raise ValueError(f"File length not divisible by reshape={reshape}: {file_path}")

    raw      = raw.reshape((reshape, raw.size // reshape), order="F")
    n_samples = raw.shape[1]
    idx      = np.arange(n_samples)

    if normalize_voltage:
        left  = ((raw[c_left,  idx] * (3.0 / (2**16)) - 1.5) / float(voltage_gain)) * 1e6
        right = ((raw[c_right, idx] * (3.0 / (2**16)) - 1.5) / float(voltage_gain)) * 1e6
    else:
        left  = raw[c_left,  idx]
        right = raw[c_right, idx]

    # Denoise each EMG channel on load (synch channel stays raw, below).
    if filter_signal:
        left  = denoise_signal(left,  sigma=sigma)
        right = denoise_signal(right, sigma=sigma)

    data = np.asarray([left, right], dtype=float)

    pulse_signal = raw[c_synch, idx]
    threshold    = pulse_signal.mean() + 3.0 * pulse_signal.std()
    above        = pulse_signal > threshold
    tms_indexes  = np.where(np.diff(above.astype(int)) == 1)[0] + 1

    return data, tms_indexes.astype(int)


# =============================================================================
# LRU-cached wrapper (avoids re-reading 100+ MB file on every rerun/drag)
# =============================================================================
@functools.lru_cache(maxsize=4)
def _load_data_cached(
    file_path_str: str,
    channels_tuple: tuple,
    fs: int,
    normalize_voltage: bool = True,
    voltage_gain: float = 133.0,
    reshape: int = 4,
    filter_signal: bool = True,
    sigma: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process-level LRU cache around load_data.
    channels_tuple: (c_synch, c_right, c_left) as plain ints.
    Returned arrays are shared — callers must NOT mutate them.
    """
    return load_data(
        file_path_str,
        channels=np.array(channels_tuple),
        fs=fs,
        normalize_voltage=normalize_voltage,
        voltage_gain=voltage_gain,
        reshape=reshape,
        filter_signal=filter_signal,
        sigma=sigma,
    )


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
    epoch: Epoch | None = None,
    pre_s: float = 0.100,
    post_s: float = 0.400,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract per-block MEP waveforms from raw EMG.
    Returns:
      t_ms:  (T,) time axis in ms relative to pulse
      meps:  (N, T) waveforms in uV
    """
    session_file = Path(session_file)
    js = read_json(session_file)

    meps_root = js.get("meps", {}) or {}
    hp        = ((meps_root.get(str(block_name)) or {}).get(str(hemi)) or {})
    pulses    = hp.get("pulses", [])
    if not isinstance(pulses, list) or len(pulses) == 0:
        return np.asarray([], dtype=float), np.zeros((0, 0), dtype=float)

    fs = float(meta.get("sampling_rate", 4000))

    input_file = meta.get("input_file", "")
    data_dir   = meta.get("data_dir", "")

    in_path = Path(str(input_file))
    if not in_path.exists() and data_dir:
        candidate = Path(str(data_dir)) / str(input_file)
        if candidate.exists():
            in_path = candidate

    channels       = meta.get("channels") or [3, 1, 2]
    channels_tuple = tuple(int(c) for c in channels)
    data, _tms    = _load_data_cached(str(in_path), channels_tuple, int(fs))

    hemi_to_ch = {"left": 0, "right": 1}
    ch  = hemi_to_ch.get(str(hemi).lower(), 0)
    sig = data[ch, :]

    if epoch is not None:
        start_ms = float(epoch.tmin_ms)
        end_ms   = float(epoch.tmax_ms)
    else:
        start_ms = -float(pre_s) * 1000.0
        end_ms   =  float(post_s) * 1000.0

    if end_ms <= start_ms:
        raise ValueError(f"Invalid window: start_ms={start_ms} must be < end_ms={end_ms}")

    start_n = int(np.round(start_ms / 1000.0 * fs))
    end_n   = int(np.round(end_ms   / 1000.0 * fs))
    n_win   = end_n - start_n
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

    return t_ms, np.asarray(trials, dtype=float)

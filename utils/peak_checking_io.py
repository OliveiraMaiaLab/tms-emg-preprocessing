"""
utils/peak_checking_io.py
-------------------------
Helpers for reading/writing peaks_flag and reading the MEP window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import json


@dataclass(frozen=True)
class Epoch:
    """Epoch bounds in milliseconds (relative to pulse)."""
    tmin_ms: float
    tmax_ms: float


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
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
    return Epoch(beg_s * 1000.0, end_s * 1000.0)


def get_peaks_flag_list(session: dict, block: str, hemi: str = "left") -> List[int]:
    meps = session.get("meps", {}) or {}
    hp = ((meps.get(block) or {}).get(hemi) or {})
    flags = hp.get("peaks_flag", [])
    if not isinstance(flags, list):
        return []
    out = []
    for v in flags:
        try:
            out.append(1 if int(v) else 0)
        except Exception:
            out.append(0)
    return out


def set_peaks_flag_list(session: dict, block: str, flags: List[int], hemi: str = "left") -> None:
    session.setdefault("meps", {})
    session["meps"].setdefault(block, {})
    session["meps"][block].setdefault(hemi, {})
    hp = session["meps"][block][hemi]
    hp["peaks_flag"] = [1 if int(v) else 0 for v in flags]
    session["meps"][block][hemi] = hp


def save_peaks_flag_list(session_file: Path, block: str, flags: List[int], hemi: str = "left") -> None:
    session = read_json(session_file)
    set_peaks_flag_list(session, block, flags, hemi=hemi)
    write_json(session_file, session)

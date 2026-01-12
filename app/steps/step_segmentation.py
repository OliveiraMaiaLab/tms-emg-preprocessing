# app/steps/step_segmentation.py
# -*- coding: utf-8 -*-
"""
steps/step_segmentation.py
--------------------
Step 3: EMG segmentation
Embeds the Bokeh viewer and writes selected ranges into the subject/session JSON.
Defensive: hydrates missing state so deep-linking works.
"""

from __future__ import annotations

import threading
import json
from pathlib import Path

import numpy as np
import streamlit as st

from app.utils.persistence import (
    ensure_metadata,
    ensure_template_loaded,
    ensure_session_file,
)
from app.bk_embedding.segmentation import start_bokeh_app
from app.utils.layout import step_nav

PREV_STEP = "input"
THIS_STEP = "segmentation"
NEXT_STEP = "mep_window"


def _segmentation_missing_flat(session_file: str, parts: list[str]) -> list[str]:
    """Return list of parts that do NOT have a valid first [start, end] in the flat schema."""
    try:
        data = json.loads(Path(session_file).read_text())
    except Exception:
        return parts[:]

    seg = data.get("segmentation", {}) or {}

    def ok(v):
        if not (isinstance(v, list) and v):
            return False
        first = v[0]
        if not (isinstance(first, (list, tuple)) and len(first) >= 2):
            return False
        try:
            s, e = float(first[0]), float(first[1])
        except (TypeError, ValueError):
            return False
        return s < e

    return [p for p in parts if not ok(seg.get(p, []))]


def _hemi_to_data_index(hemi: str) -> int:
    """
    load_data() returns data as [left, right].
    """
    h = str(hemi).strip().lower()
    if h == "right":
        return 1
    return 0  # default left


def _compute_rest_ref_from_emg_ref(seg: dict, emg: np.ndarray, fs: float) -> float | None:
    """
    If segmentation has 'emg_ref': [[start_s, end_s]], compute:
      rest_ref = std(abs(emg[first 0.05s of that segment]))
    Returns None if emg_ref is missing/invalid.
    """
    v = seg.get("emg_ref", [])
    if not (isinstance(v, list) and v and isinstance(v[0], (list, tuple)) and len(v[0]) >= 2):
        return None

    try:
        start_s = float(v[0][0])
    except Exception:
        return None

    t_interval = 0.05
    a = int(round(start_s * fs))
    b = int(round((start_s + t_interval) * fs))

    a = max(0, min(a, emg.size))
    b = max(0, min(b, emg.size))

    if b <= a:
        return None

    return float(np.std(np.abs(emg[a:b])))


def _std_muscle_activity_flag(pulse_idx: int, emg: np.ndarray, rest_ref: float, fs: float = 4000.0) -> int:
    """
    Fills std_preactivation_flag.
    Uses 0.05s window ending 5 samples before pulse:
      samples = [pulse-5-0.05*fs, pulse-5)
      pre_pulse = std(abs(emg[samples]))
      return 1 if pre_pulse > rest_ref else 0
    """
    t_interval = 0.05
    end = int(pulse_idx - 5)
    beg = int(end - t_interval * fs)

    if end <= 0:
        return 0
    beg = max(0, beg)
    end = min(emg.size, end)
    if end <= beg:
        return 0

    pre_pulse = float(np.std(np.abs(emg[beg:end])))
    return 1 if pre_pulse > rest_ref else 0


def _hinder_muscle_activity_flag(pulse_idx: int, emg: np.ndarray, fs: float = 4000.0) -> int:
    """
    Fills hinder_preactivation_flag.
    Same window:
      rest_ref = 15
      return 1 if std(abs(pre)) > rest_ref else 0
    """
    rest_ref = 15
    t_interval = 0.05
    end = int(pulse_idx - 5)
    beg = int(end - t_interval * fs)

    if end <= 0:
        return 0
    beg = max(0, beg)
    end = min(emg.size, end)
    if end <= beg:
        return 0

    pre_pulse = float(np.std(np.abs(emg[beg:end])))
    return 1 if pre_pulse > rest_ref else 0


def compute_and_store_mep_pulses(session_file: str, meta: dict, blocks: list[str]):
    """
    Writes pulse indices (sample indices) into:
      meps.<block>.<hemi>.pulses
    and initializes/fills parallel lists:
      min, max, hinder_preactivation_flag, std_preactivation_flag, peaks_flag

    NOTE: min/max stay None here (computed later in MEP window / overlap step).
    """
    from app.utils.tms_module import load_data

    js = json.loads(Path(session_file).read_text())
    info = js.get("info") or {}

    data_file = meta.get("input_file") or info.get("input_file")
    if not data_file:
        raise RuntimeError("No input_file found in meta or session_file['info'].")

    fs = float(meta["sampling_rate"])

    # IMPORTANT: use meta["channels"] so the sync pulse channel is correct
    data, tms_indexes = load_data(data_file, channels=meta["channels"], fs=int(fs))
    tms_indexes = np.asarray(tms_indexes, dtype=int)
    pulse_times_s = tms_indexes / fs

    seg = js.get("segmentation") or {}
    if not isinstance(seg, dict):
        raise RuntimeError("session_file['segmentation'] is not a dict.")

    hemis = list(meta.get("hemispheres", info.get("hemispheres", ["left"])))

    meps_root = js.get("meps")
    if not isinstance(meps_root, dict):
        meps_root = {}

    # Precompute rest_ref per hemi if emg_ref exists
    rest_ref_by_hemi: dict[str, float | None] = {}
    for h in hemis:
        emg = np.asarray(data[_hemi_to_data_index(h)], dtype=float)
        rest_ref_by_hemi[h] = _compute_rest_ref_from_emg_ref(seg, emg, fs)

    for block in blocks:
        seg_v = seg.get(block, [])
        if not (
            isinstance(seg_v, list)
            and seg_v
            and isinstance(seg_v[0], (list, tuple))
            and len(seg_v[0]) >= 2
        ):
            meps_root.setdefault(block, {})
            for h in hemis:
                meps_root[block][h] = {
                    "pulses": [],
                    "min": [],
                    "max": [],
                    "hinder_preactivation_flag": [],
                    "std_preactivation_flag": [],
                    "peaks_flag": [],
                }
            continue

        seg_s, seg_e = map(float, seg_v[0])
        keep = (pulse_times_s >= seg_s) & (pulse_times_s <= seg_e)
        pulses = tms_indexes[keep].astype(int).tolist()
        n = len(pulses)

        meps_root.setdefault(block, {})
        for h in hemis:
            emg = np.asarray(data[_hemi_to_data_index(h)], dtype=float)

            hinder_flags = [_hinder_muscle_activity_flag(p, emg, fs=fs) for p in pulses]

            rr = rest_ref_by_hemi.get(h)
            if rr is None:
                std_flags = [0] * n
            else:
                std_flags = [_std_muscle_activity_flag(p, emg, rr, fs=fs) for p in pulses]

            meps_root[block][h] = {
                "pulses": pulses,
                "min": [None] * n,
                "max": [None] * n,
                "hinder_preactivation_flag": hinder_flags,
                "std_preactivation_flag": std_flags,
                "peaks_flag": [0] * n,
            }

    js["meps"] = meps_root
    Path(session_file).write_text(json.dumps(js, indent=2))


def run_step(meta: dict):
    meta = ensure_metadata()
    meta = ensure_template_loaded(meta)
    session_file = ensure_session_file(meta)

    if "_ranges_store" not in st.session_state:
        st.session_state["_ranges_store"] = {}
    if "_ranges_lock" not in st.session_state:
        st.session_state["_ranges_lock"] = threading.Lock()

    def _try_advance() -> bool:
        required_parts = list(meta.get("exp_structure", []))

        missing = _segmentation_missing_flat(session_file, required_parts)
        if missing:
            st.session_state["_segmentation_incomplete"] = True
            st.session_state["_segmentation_missing"] = missing
            return False

        mep_blocks = ["bmeps", "t0meps", "t10meps", "t20meps", "t30meps"]
        mep_blocks = [b for b in mep_blocks if b in required_parts]

        try:
            compute_and_store_mep_pulses(session_file=session_file, meta=meta, blocks=mep_blocks)
        except Exception as e:
            st.toast(f"Failed to compute/store MEP pulses: {e}", icon="❌")
            return False

        return True

    step_nav(
        THIS_STEP,
        step_title="EMG Segmentation",
        back_step=PREV_STEP,
        next_step=NEXT_STEP,
        on_next=_try_advance,
        right_label="Advance ▶",
    )

    if st.session_state.pop("_segmentation_incomplete", False):
        missing = st.session_state.pop("_segmentation_missing", [])
        msg = "Finish segmentation before advancing"
        if missing:
            msg += f": {', '.join(missing)}"
        try:
            st.toast(msg, icon="⚠️")
        except Exception:
            st.warning(msg)

    bokeh_key = (
        session_file,
        meta.get("input_file"),
        meta.get("template_file"),
        tuple(meta.get("hemispheres", [])),
        tuple(meta.get("exp_structure", [])),
    )
    if st.session_state.get("_bokeh_key") != bokeh_key:
        st.session_state["_bokeh_key"] = bokeh_key
        st.session_state["_bokeh_port"] = start_bokeh_app(
            meta=meta,
            session_file=session_file,
            exp_structure=meta["exp_structure"],
            SCRIPT_DIR=meta.get("_script_dir", "."),
            ranges_store=st.session_state["_ranges_store"],
            ranges_lock=st.session_state["_ranges_lock"],
        )

    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] { overflow-x: hidden; }
        .bk-iframe-wrap { width: 100%; overflow-x: hidden; }
        .bk-iframe-wrap iframe { display: block; width: 100%; border: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    iframe_height = 550 * max(1, len(meta["hemispheres"]))
    bokeh_url = f"http://localhost:{st.session_state['_bokeh_port']}/bkapp"

    st.markdown(
        f"""
        <div class="bk-iframe-wrap">
            <iframe src="{bokeh_url}" height="{iframe_height}" scrolling="no"></iframe>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(f"Session file: `{session_file}`")

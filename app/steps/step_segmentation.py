# app/steps/step_segmentation.py
# -*- coding: utf-8 -*-
"""
steps/step_segmentation.py
--------------------------
Step 3: EMG Segmentation.

Block management:
  - All blocks (minus emg_ref) can be activated/deactivated via a multiselect.
  - An "Update" button persists the selection to JSON, clears segmentation ranges
    for deactivated blocks, and restarts the Bokeh server with the filtered list.

Downsampling:
  - File duration and raw point count are estimated from file size.
  - Default downsample factor is computed to keep the Bokeh plot under
    ~200 k points per channel; factor=1 means no downsampling.
  - The user can override the factor and click "Update" to re-render.
  - The Bokeh server restarts any time active_blocks or the factor changes.
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
    _manageable_blocks,
)
from app.bk_embedding.segmentation import start_bokeh_app
from app.utils.layout import step_nav

PREV_STEP = "input"
THIS_STEP = "segmentation"
NEXT_STEP = "mep_window"

_SAFE_POINTS = 200_000   # target max points per channel in Bokeh
_RESHAPE     = 4         # channels in file (uint16 interleaved)


def _estimate_file_info(input_file: str, fs: float) -> tuple[int, float, int]:
    """
    Estimate (n_samples, duration_s, suggested_factor) from file size.
    No actual loading — purely from os.stat.
    """
    try:
        file_bytes = Path(input_file).stat().st_size
        n_samples  = file_bytes // (_RESHAPE * 2)   # uint16 = 2 bytes
        duration_s = n_samples / fs
        factor     = max(1, n_samples // _SAFE_POINTS)
        return n_samples, duration_s, factor
    except Exception:
        return 0, 0.0, 1


def _read_active_blocks(session_file: str, exp_structure: list[str]) -> list[str]:
    try:
        js   = json.loads(Path(session_file).read_text())
        saved = js.get("active_blocks")
        mgbl = _manageable_blocks(exp_structure)
        if isinstance(saved, list):
            return [b for b in mgbl if b in set(saved)]
        return list(mgbl)
    except Exception:
        return _manageable_blocks(exp_structure)


def _write_active_blocks(
    session_file: str,
    new_active: list[str],
    clear_segmentation_for: list[str],
) -> None:
    js  = json.loads(Path(session_file).read_text())
    js["active_blocks"] = new_active
    seg = js.get("segmentation") or {}
    for b in clear_segmentation_for:
        if b in seg:
            seg[b] = []
    js["segmentation"] = seg
    Path(session_file).write_text(json.dumps(js, indent=2))


def _segmentation_missing_flat(session_file: str, parts: list[str]) -> list[str]:
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
    return 1 if str(hemi).strip().lower() == "right" else 0


def _compute_rest_ref_from_emg_ref(seg: dict, emg: np.ndarray, fs: float) -> float | None:
    v = seg.get("emg_ref", [])
    if not (isinstance(v, list) and v and isinstance(v[0], (list, tuple)) and len(v[0]) >= 2):
        return None
    try:
        start_s = float(v[0][0])
    except Exception:
        return None
    t_interval = 0.05
    a = max(0, int(round(start_s * fs)))
    b = max(0, min(emg.size, int(round((start_s + t_interval) * fs))))
    return float(np.std(np.abs(emg[a:b]))) if b > a else None


def _std_muscle_activity_flag(pulse_idx: int, emg: np.ndarray, rest_ref: float, fs: float = 4000.0) -> int:
    t_interval = 0.05
    end = int(pulse_idx - 5)
    beg = max(0, int(end - t_interval * fs))
    if end <= 0:
        return 0
    end = min(emg.size, end)
    return 1 if (end > beg and float(np.std(np.abs(emg[beg:end]))) > rest_ref) else 0


def _hinder_muscle_activity_flag(pulse_idx: int, emg: np.ndarray, fs: float = 4000.0) -> int:
    rest_ref   = 15
    t_interval = 0.05
    end = int(pulse_idx - 5)
    beg = max(0, int(end - t_interval * fs))
    if end <= 0:
        return 0
    end = min(emg.size, end)
    return 1 if (end > beg and float(np.std(np.abs(emg[beg:end]))) > rest_ref) else 0


def compute_and_store_mep_pulses(session_file: str, meta: dict, blocks: list[str]):
    from app.utils.tms_module import _load_data_cached

    js   = json.loads(Path(session_file).read_text())
    info = js.get("info") or {}

    data_file = meta.get("input_file") or info.get("input_file")
    if not data_file:
        raise RuntimeError("No input_file found.")

    fs             = float(meta["sampling_rate"])
    channels       = meta.get("channels") or [3, 1, 2]
    channels_tuple = tuple(int(c) for c in channels)

    data, tms_indexes = _load_data_cached(str(data_file), channels_tuple, int(fs))
    tms_indexes   = np.asarray(tms_indexes, dtype=int)
    pulse_times_s = tms_indexes / fs

    seg   = js.get("segmentation") or {}
    hemis = list(meta.get("hemispheres", info.get("hemispheres", ["left"])))
    meps_root = js.get("meps") if isinstance(js.get("meps"), dict) else {}

    rest_ref_by_hemi: dict[str, float | None] = {}
    for h in hemis:
        emg = np.asarray(data[_hemi_to_data_index(h)], dtype=float)
        rest_ref_by_hemi[h] = _compute_rest_ref_from_emg_ref(seg, emg, fs)

    for block in blocks:
        seg_v = seg.get(block, [])
        if not (isinstance(seg_v, list) and seg_v
                and isinstance(seg_v[0], (list, tuple)) and len(seg_v[0]) >= 2):
            meps_root.setdefault(block, {})
            for h in hemis:
                meps_root[block][h] = {
                    "pulses": [], "min": [], "max": [],
                    "hinder_preactivation_flag": [], "std_preactivation_flag": [],
                    "peaks_flag": [], "noise_flag": [], "below_threshold_flag": [],
                }
            continue

        seg_s, seg_e = map(float, seg_v[0])
        keep   = (pulse_times_s >= seg_s) & (pulse_times_s <= seg_e)
        pulses = tms_indexes[keep].astype(int).tolist()
        n      = len(pulses)

        meps_root.setdefault(block, {})
        for h in hemis:
            emg          = np.asarray(data[_hemi_to_data_index(h)], dtype=float)
            hinder_flags = [_hinder_muscle_activity_flag(p, emg, fs=fs) for p in pulses]
            rr           = rest_ref_by_hemi.get(h)
            std_flags    = [_std_muscle_activity_flag(p, emg, rr, fs=fs) for p in pulses] if rr else [0] * n

            meps_root[block][h] = {
                "pulses":                    pulses,
                "min":                       [None] * n,
                "max":                       [None] * n,
                "hinder_preactivation_flag": hinder_flags,
                "std_preactivation_flag":    std_flags,
                "peaks_flag":                [0] * n,
                "noise_flag":                [0] * n,
                "below_threshold_flag":      [0] * n,
            }

    js["meps"] = meps_root
    Path(session_file).write_text(json.dumps(js, indent=2))


def run_step(meta: dict):
    meta         = ensure_metadata()
    meta         = ensure_template_loaded(meta)
    session_file = ensure_session_file(meta)

    if "_ranges_store" not in st.session_state:
        st.session_state["_ranges_store"] = {}
    if "_ranges_lock" not in st.session_state:
        st.session_state["_ranges_lock"] = threading.Lock()

    # ---- file info (from size, no loading needed) ----
    fs = float(meta.get("sampling_rate", 4000))
    n_samples, duration_s, default_factor = _estimate_file_info(
        meta.get("input_file", ""), fs
    )

    # ---- read current active blocks ----
    all_parts  = list(meta.get("exp_structure", []))
    manageable = _manageable_blocks(all_parts)
    saved_active = _read_active_blocks(session_file, all_parts)

    # ---- block + downsample controls ----
    with st.expander("⚙️ Signal display & block settings", expanded=False):
        st.markdown("**Active blocks**")
        st.caption(
            "Deselect blocks not present in this recording — they will be skipped "
            "during segmentation validation and peak checking. "
            "`emg_ref` is always required and cannot be removed."
        )

        selected_blocks = st.multiselect(
            "Active blocks",
            options=manageable,
            default=saved_active,
            key="_seg_active_blocks_input",
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**Signal display**")

        if n_samples > 0:
            st.caption(
                f"File: **{duration_s/60:.1f} min** ({n_samples:,} samples at {int(fs)} Hz). "
                f"Suggested factor for fast rendering: **{default_factor}** "
                f"(factor 1 = no downsampling, full resolution)."
            )
        else:
            st.caption("Could not read file size — default factor 1.")

        downsample_factor = st.number_input(
            "Downsample factor (1 = full resolution)",
            min_value=1,
            max_value=max(1, n_samples // 1000) if n_samples > 0 else 1000,
            value=int(st.session_state.get("_seg_ds_factor", default_factor)),
            step=1,
            key="_seg_ds_factor_input",
            help="Higher values skip more samples and render faster. "
                 "Use 1 for full resolution; increase if the plot lags.",
        )

        if st.button("✅ Update plot & block list", type="primary"):
            # Persist active_blocks to JSON
            newly_deactivated = [b for b in saved_active if b not in set(selected_blocks)]
            _write_active_blocks(
                session_file,
                new_active=[b for b in manageable if b in set(selected_blocks)],
                clear_segmentation_for=newly_deactivated,
            )
            # Store the chosen factor
            st.session_state["_seg_ds_factor"] = int(downsample_factor)
            # Force Bokeh server restart on next render
            st.session_state.pop("_bokeh_key", None)
            st.rerun()

    # Use the last confirmed values (only updated on "Update" click).
    confirmed_active  = _read_active_blocks(session_file, all_parts)
    confirmed_factor  = int(st.session_state.get("_seg_ds_factor", default_factor))

    # ---- advance logic ----
    def _try_advance() -> bool:
        js_inner     = json.loads(Path(session_file).read_text())
        cur_active   = set(js_inner.get("active_blocks", manageable))
        required     = [b for b in all_parts if b in cur_active or b == "emg_ref"]

        missing = _segmentation_missing_flat(session_file, required)
        if missing:
            st.session_state["_segmentation_incomplete"] = True
            st.session_state["_segmentation_missing"]    = missing
            return False

        mep_blocks = [b for b in required if str(b).lower().endswith("meps")]
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

    # ---- Bokeh server (restart when active_blocks or factor changes) ----
    bokeh_key = (
        session_file,
        meta.get("input_file"),
        meta.get("template_file"),
        tuple(meta.get("hemispheres", [])),
        tuple(confirmed_active),     # restart when block selection changes
        confirmed_factor,             # restart when downsample factor changes
    )
    if st.session_state.get("_bokeh_key") != bokeh_key:
        st.session_state["_bokeh_key"]  = bokeh_key
        st.session_state["_bokeh_port"] = start_bokeh_app(
            meta=meta,
            session_file=session_file,
            exp_structure=meta["exp_structure"],
            active_blocks=confirmed_active,
            downsample_factor=confirmed_factor,
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
    bokeh_url     = f"http://localhost:{st.session_state['_bokeh_port']}/bkapp"
    st.markdown(
        f"""
        <div class="bk-iframe-wrap">
            <iframe src="{bokeh_url}" height="{iframe_height}" scrolling="no"></iframe>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Session file: `{session_file}` · "
        f"Displaying: {', '.join(confirmed_active) or '(none)'} · "
        f"Downsample factor: {confirmed_factor}"
    )

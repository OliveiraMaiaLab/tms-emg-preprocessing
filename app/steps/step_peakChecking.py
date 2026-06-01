# app/steps/step_peakChecking.py
# -*- coding: utf-8 -*-
"""
steps/step_peakChecking.py
--------------------------
Step 5: Peak checking.

Two flags per MEP:
  noise_flag  — signal is unusable; checking this forces peaks_flag to 0
                and disables it in the UI.
  peaks_flag  — "Incorrect peak" — needs manual correction in step 6.

Bulk noise controls (per page): Select all / Deselect all / Toggle all.
Only active MEP blocks (from session["active_blocks"]) are shown.
"""

from __future__ import annotations

from pathlib import Path
from math import ceil
import io

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FuncFormatter

from app.utils.persistence import (
    ensure_metadata,
    ensure_template_loaded,
    ensure_session_file,
)
from app.utils.layout import step_nav
from app.utils.tms_module import (
    load_meps_for_block,
    Epoch,
    read_json,
    get_epoch_from_session,
    get_flag_list,
    save_flag_list,
)


plt.rcParams.update({
    "font.size":         22,
    "axes.titlesize":    26,
    "axes.labelsize":    24,
    "xtick.labelsize":   20,
    "ytick.labelsize":   20,
    "legend.fontsize":   20,
    "lines.linewidth":   3.0,
    "lines.markersize":  10,
    "axes.linewidth":    2.0,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.major.size":  8,
    "ytick.major.size":  8,
    "figure.dpi":        96,
})

PREV_STEP = "mep_window"
THIS_STEP = "peak_checking"
NEXT_STEP = "peak_correction"

FLAG_NOISE = "noise_flag"
FLAG_PEAKS = "peaks_flag"

WORK_PREFIX_NOISE = "noise_flag_work"
WORK_PREFIX_PEAKS = "peaks_flag_work"

_ALL_FLAG_CONFIGS = [
    (WORK_PREFIX_PEAKS, FLAG_PEAKS),
    (WORK_PREFIX_NOISE, FLAG_NOISE),
]


# -------------------------------------------------------------------
# PNG cache helpers
# -------------------------------------------------------------------
def _pc_png_cache_root_key(block: str, hemi: str) -> str:
    return f"_pc_png_cache::{block}::{hemi}"

def _get_png_cache(block: str, hemi: str) -> dict:
    key = _pc_png_cache_root_key(block, hemi)
    if key not in st.session_state or not isinstance(st.session_state[key], dict):
        st.session_state[key] = {}
    return st.session_state[key]

def _png_key(mep_idx: int, mep_min, mep_max) -> tuple:
    def _freeze(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple)) and len(x) == 2:
            a, b = x
            return (None if a is None else float(a), None if b is None else float(b))
        return (str(type(x)), str(x))
    return (int(mep_idx), _freeze(mep_min), _freeze(mep_max))

def _clear_png_cache(block: str, hemi: str) -> None:
    st.session_state.pop(_pc_png_cache_root_key(block, hemi), None)


# -------------------------------------------------------------------
# Seen-mask helpers
# -------------------------------------------------------------------
def _pc_seen_mask_key(block: str, hemi: str) -> str:
    return f"_pc_seen_mask::{block}::{hemi}"

def _pc_nmeps_cache_key(hemi: str) -> str:
    return f"_pc_nmeps_cache::{hemi}"

def _pc_all_blocks_key() -> str:
    return "_pc_all_blocks"

def _pc_hemi_key() -> str:
    return "_pc_hemi"


def _ensure_seen_mask(block: str, hemi: str, n_meps: int) -> list[int]:
    key  = _pc_seen_mask_key(block, hemi)
    mask = st.session_state.get(key)
    if not isinstance(mask, list) or len(mask) != n_meps:
        mask = ([0] * n_meps if not isinstance(mask, list)
                else (mask + [0] * n_meps)[:n_meps])
        st.session_state[key] = mask
    return mask

def _mark_meps_seen(block: str, hemi: str, n_meps: int, mep_indices: list[int]) -> None:
    mask = _ensure_seen_mask(block, hemi, n_meps)
    for idx in mep_indices:
        if 0 <= idx < len(mask):
            mask[idx] = 1
    st.session_state[_pc_seen_mask_key(block, hemi)] = mask

def _cache_n_meps(block: str, hemi: str, n_meps: int) -> None:
    key = _pc_nmeps_cache_key(hemi)
    if key not in st.session_state or not isinstance(st.session_state[key], dict):
        st.session_state[key] = {}
    st.session_state[key][str(block)] = int(n_meps)

def _get_cached_n_meps(block: str, hemi: str) -> int | None:
    cache = st.session_state.get(_pc_nmeps_cache_key(hemi))
    if not isinstance(cache, dict):
        return None
    val = cache.get(str(block))
    return int(val) if isinstance(val, int) else None

def _seen_count_for_block(block: str, hemi: str, n_meps: int) -> tuple[int, int]:
    mask = _ensure_seen_mask(block, hemi, n_meps)
    return (sum(mask), n_meps)

def _all_seen_for_block(block: str, hemi: str, n_meps: int) -> bool:
    return all(v == 1 for v in _ensure_seen_mask(block, hemi, n_meps))


# -------------------------------------------------------------------
# Work-list helpers
# -------------------------------------------------------------------
def _load_work_list(session: dict, block: str, hemi: str, flag_name: str,
                    work_prefix: str, n_meps: int) -> list[int]:
    key = f"{work_prefix}::{block}::{hemi}"
    if key not in st.session_state:
        flags = get_flag_list(session, block, hemi, flag_name)
        if len(flags) != n_meps:
            flags = (flags + [0] * n_meps)[:n_meps]
        st.session_state[key] = flags
    return st.session_state[key]


def _autosave_prev_if_nav_changed(session_file: Path) -> None:
    prev = st.session_state.get("_pc_prev")
    cur  = st.session_state.get("_pc_cur")

    if prev is None or cur is None or prev == cur:
        return

    prev_block = prev["block"]
    prev_hemi  = prev.get("hemi", "left")

    if prev["block"] != cur["block"] or prev.get("hemi") != cur.get("hemi"):
        _clear_png_cache(block=prev_block, hemi=prev_hemi)

    any_saved = False
    for work_prefix, flag_name in _ALL_FLAG_CONFIGS:
        work_key = f"{work_prefix}::{prev_block}::{prev_hemi}"
        if work_key in st.session_state:
            save_flag_list(session_file, prev_block,
                           st.session_state[work_key], prev_hemi, flag_name)
            any_saved = True

    if any_saved:
        st.toast("Auto-saved flags", icon="💾")


# -------------------------------------------------------------------
# Per-MEP cell renderer
# -------------------------------------------------------------------
def plot_mep_cell(
    t_ms: np.ndarray,
    y_data: np.ndarray,
    mep_idx: int,
    noise_flags: list,
    peaks_flags: list,
    epoch: Epoch,
    block: str,
    hemi: str = "left",
    mep_min: tuple | None = None,
    mep_max: tuple | None = None,
) -> None:
    # ---- waveform image (cached) ----
    cache = _get_png_cache(block=block, hemi=hemi)
    k     = _png_key(mep_idx, mep_min, mep_max)

    if k not in cache:
        fig, ax = plt.subplots()

        if mep_min is not None:
            min_ms, min_val = mep_min
            if min_ms is not None and min_val is not None:
                ax.scatter([min_ms], [min_val], marker=6, s=250)
        if mep_max is not None:
            max_ms, max_val = mep_max
            if max_ms is not None and max_val is not None:
                ax.scatter([max_ms], [max_val], marker=7, s=250)

        ax.plot(t_ms, y_data)
        ymin, ymax = ax.get_ylim()
        pad = 0.05 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.yaxis.set_major_locator(LinearLocator(5))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:8.0f}"))
        ax.set_xticks([])
        ax.margins(0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        cache[k] = buf.getvalue()

    st.image(cache[k], use_container_width=True)
    st.caption(f"MEP {mep_idx}")

    # ---- checkboxes ----
    is_noise = st.checkbox(
        "Noise",
        value=bool(noise_flags[mep_idx]),
        key=f"chk_noise::{block}::{hemi}::{mep_idx}",
    )

    # Noise checked → force Incorrect peak to 0 and disable it.
    is_incorrect = st.checkbox(
        "Incorrect peak",
        value=bool(peaks_flags[mep_idx]) and not is_noise,
        key=f"chk_peak::{block}::{hemi}::{mep_idx}",
        disabled=is_noise,
    )

    noise_flags[mep_idx] = 1 if is_noise else 0
    peaks_flags[mep_idx] = 0 if is_noise else (1 if is_incorrect else 0)


# -------------------------------------------------------------------
# Main step
# -------------------------------------------------------------------
def run_step(meta: dict):
    meta         = ensure_metadata()
    meta         = ensure_template_loaded(meta)
    session_file = Path(ensure_session_file(meta))

    def _on_next() -> bool:
        cur = st.session_state.get("_pc_cur")
        if cur:
            for work_prefix, flag_name in _ALL_FLAG_CONFIGS:
                work_key = f"{work_prefix}::{cur['block']}::{cur.get('hemi', 'left')}"
                if work_key in st.session_state:
                    save_flag_list(session_file, cur["block"],
                                   st.session_state[work_key],
                                   cur.get("hemi", "left"), flag_name)

        all_blocks = st.session_state.get(_pc_all_blocks_key(), [])
        hemis      = list(meta.get("hemispheres", ["left"]))

        if not all_blocks:
            st.error("Internal error: no MEP blocks found in state.")
            return False

        not_visited = []
        incomplete  = []

        for h in hemis:
            for b in all_blocks:
                n_b = _get_cached_n_meps(b, h)
                if n_b is None:
                    not_visited.append(f"{b}/{h}")
                    continue
                if n_b <= 0:
                    continue
                if not _all_seen_for_block(b, h, n_b):
                    seen_b, total_b = _seen_count_for_block(b, h, n_b)
                    incomplete.append((f"{b}/{h}", seen_b, total_b))

        if not_visited or incomplete:
            msg = "You must plot all MEPs at least once in **all blocks and hemispheres** before advancing.\n\n"
            if not_visited:
                msg += "Not visited yet:\n" + "\n".join(f"- {bh}" for bh in not_visited) + "\n\n"
            if incomplete:
                msg += "Still incomplete:\n" + "\n".join(
                    f"- {bh}: {s}/{t}" for bh, s, t in incomplete
                )
            st.toast(msg)
            return False

        return True

    step_nav(
        THIS_STEP,
        step_title="Peak Checking",
        back_step=PREV_STEP,
        next_step=NEXT_STEP,
        on_next=_on_next,
        disabled_next=False,
    )

    session = read_json(session_file)
    epoch   = get_epoch_from_session(session)

    active_set = set(session.get("active_blocks", meta.get("exp_structure", [])))
    blocks     = [
        b for b in (meta.get("exp_structure", []) or [])
        if str(b).lower().endswith("meps") and b in active_set
    ]
    st.session_state[_pc_all_blocks_key()] = [str(b) for b in blocks]

    if not blocks:
        st.error("No active '*meps' blocks found. Check the block selection in the Segmentation step.")
        return

    hemis     = list(meta.get("hemispheres", ["left"]))
    n_ctrl    = 5 if len(hemis) > 1 else 4
    ctrl_cols = st.columns([1] * n_ctrl)

    with ctrl_cols[0]:
        block = st.selectbox("Block", blocks, index=0, key="_pc_block_sel")

    col_off = 1
    if len(hemis) > 1:
        with ctrl_cols[1]:
            hemi = st.selectbox("Hemisphere", hemis, index=0, key="_pc_hemi_sel")
        col_off = 2
    else:
        hemi = hemis[0]

    with ctrl_cols[col_off]:
        n_rows = st.number_input("Rows", min_value=1, max_value=12, value=4, step=1, key="_pc_rows_sel")
    with ctrl_cols[col_off + 1]:
        n_cols = st.number_input("Cols", min_value=1, max_value=12, value=10, step=1, key="_pc_cols_sel")

    st.session_state[_pc_hemi_key()] = hemi
    page_size = int(n_rows) * int(n_cols)

    t_ms, meps = load_meps_for_block(
        meta, session_file, block_name=block, hemi=hemi, epoch=epoch, detrend=False
    )
    if meps.size == 0 or meps.ndim != 2:
        st.warning("No MEPs found for this block (missing pulses or extraction failed).")
        return

    n_meps  = int(meps.shape[0])
    n_pages = int(ceil(n_meps / max(1, page_size)))
    _cache_n_meps(str(block), hemi, n_meps)

    with ctrl_cols[col_off + 2]:
        page = st.number_input(
            "Page", min_value=1, max_value=max(1, n_pages),
            value=1, step=1, key="_pc_page_sel",
        )

    # ---- autosave tracking ----
    st.session_state["_pc_prev"] = st.session_state.get("_pc_cur")
    st.session_state["_pc_cur"]  = {
        "block": str(block), "hemi": str(hemi),
        "page": int(page), "rows": int(n_rows), "cols": int(n_cols),
    }
    _autosave_prev_if_nav_changed(session_file)

    # ---- load work lists ----
    session     = read_json(session_file)
    noise_flags = _load_work_list(session, str(block), hemi, FLAG_NOISE, WORK_PREFIX_NOISE, n_meps)
    peaks_flags = _load_work_list(session, str(block), hemi, FLAG_PEAKS, WORK_PREFIX_PEAKS, n_meps)

    hemi_payload = (((session.get("meps") or {}).get(str(block)) or {}).get(hemi) or {})
    mins = hemi_payload.get("min", [])
    maxs = hemi_payload.get("max", [])

    # ---- paging ----
    start       = (int(page) - 1) * page_size
    end         = min(n_meps, start + page_size)
    mep_indices = list(range(start, end))

    st.caption(f"Showing MEPs **{start}–{end-1}** of **{n_meps}** (0-indexed).")
    seen, total = _seen_count_for_block(str(block), hemi, n_meps)
    st.caption(f"Reviewed in this block/hemi: **{seen}/{total}**")
    _mark_meps_seen(str(block), hemi, n_meps, mep_indices)

    # ---- bulk noise controls (per page) ----
    st.markdown("**Noise — this page:**")

    # Tighten checkbox column gaps inside MEP grid cells.
    st.markdown(
        """
        <style>
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            padding-left: 0 !important;
            padding-right: 0.25rem !important;
            min-width: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    bn_sel, bn_desel, bn_toggle = st.columns(3)

    with bn_sel:
        if st.button("Select all", key=f"_noise_selall::{block}::{hemi}::{page}"):
            for idx in mep_indices:
                noise_flags[idx] = 1
                peaks_flags[idx] = 0
                st.session_state[f"chk_noise::{block}::{hemi}::{idx}"] = True
                st.session_state[f"chk_peak::{block}::{hemi}::{idx}"]  = False
            st.rerun()

    with bn_desel:
        if st.button("Deselect all", key=f"_noise_deselall::{block}::{hemi}::{page}"):
            for idx in mep_indices:
                noise_flags[idx] = 0
                st.session_state[f"chk_noise::{block}::{hemi}::{idx}"] = False
            st.rerun()

    with bn_toggle:
        if st.button("Toggle all", key=f"_noise_toggleall::{block}::{hemi}::{page}"):
            for idx in mep_indices:
                new_val = 1 - noise_flags[idx]
                noise_flags[idx] = new_val
                if new_val == 1:
                    peaks_flags[idx] = 0
                    st.session_state[f"chk_peak::{block}::{hemi}::{idx}"]  = False
                st.session_state[f"chk_noise::{block}::{hemi}::{idx}"] = bool(new_val)
            st.rerun()

    # ---- MEP grid ----
    for i, mep_idx in enumerate(mep_indices):
        c = i % int(n_cols)
        if c == 0:
            columns = st.columns(int(n_cols))

        with columns[c]:
            mep_min = mins[mep_idx] if mep_idx < len(mins) else None
            mep_max = maxs[mep_idx] if mep_idx < len(maxs) else None

            plot_mep_cell(
                t_ms=t_ms,
                y_data=meps[mep_idx],
                mep_idx=mep_idx,
                noise_flags=noise_flags,
                peaks_flags=peaks_flags,
                epoch=epoch,
                block=str(block),
                hemi=hemi,
                mep_min=mep_min,
                mep_max=mep_max,
            )

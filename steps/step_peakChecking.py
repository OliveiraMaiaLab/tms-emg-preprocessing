"""
steps/step_peakChecking.py
--------------------------
Step 5: Peak checking

- Select a block
- Plot one axis per MEP (grid)
- Mark MEP window bounds (vertical lines)
- Checkbox per MEP to set peaks_flag (0/1)
- Auto-save peaks_flag when block / page / layout changes
- Persist into session["meps"][block]["left"]["peaks_flag"]
"""

from __future__ import annotations

from pathlib import Path
from math import ceil

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FuncFormatter



from utils.persistence import (
    ensure_metadata,
    ensure_template_loaded,
    ensure_session_file,
)
from utils.layout import render_text, step_nav
from utils.tms_module import (
    load_meps_for_block,
    Epoch,
    read_json,
    get_epoch_from_session,
    get_peaks_flag_list,
    save_peaks_flag_list,  # <-- your function: (session_file, block, flags, hemi)
)

plt.rcParams.update({
    # Fonts
    "font.size": 22,
    "axes.titlesize": 26,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,

    # Lines / markers
    "lines.linewidth": 3.0,
    "lines.markersize": 10,

    # Axes
    "axes.linewidth": 2.0,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.major.size": 8,
    "ytick.major.size": 8,

    # Resolution
    "figure.dpi": 30,
})

PREV_STEP = "mep_window"
THIS_STEP = "peak_checking"
NEXT_STEP = "peak_correction"

import io


def _freeze_extrema(x):
    """
    Convert extrema (tuple/list like [ms, val]) into a hashable (ms, val) tuple.
    Returns None if x is None.
    """
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and len(x) == 2:
        a, b = x
        # keep None as None; cast numbers to float for stability
        a = None if a is None else float(a)
        b = None if b is None else float(b)
        return (a, b)
    # unexpected shape/type -> make it hashable anyway
    return (str(type(x)), str(x))


def _pc_png_cache_root_key(block: str, hemi: str) -> str:
    return f"_pc_png_cache::{block}::{hemi}"

def _get_png_cache(block: str, hemi: str) -> dict:
    key = _pc_png_cache_root_key(block, hemi)
    if key not in st.session_state or not isinstance(st.session_state[key], dict):
        st.session_state[key] = {}
    return st.session_state[key]

def _png_key(mep_idx: int, mep_min, mep_max) -> tuple:
    """
    Cache key for a rendered MEP image.
    Include anything that would change the visual output.
    """
    # mep_min / mep_max are tuples like (ms, val) or None
    return (int(mep_idx), _freeze_extrema(mep_min), _freeze_extrema(mep_max))

def _clear_png_cache(block: str, hemi: str) -> None:
    key = _pc_png_cache_root_key(block, hemi)
    st.session_state.pop(key, None)

def _pc_seen_mask_key(block: str, hemi: str) -> str:
    return f"_pc_seen_mask::{block}::{hemi}"

def _pc_nmeps_cache_key(hemi: str) -> str:
    return f"_pc_nmeps_cache::{hemi}"  # dict: {block: n_meps}

def _pc_all_blocks_key() -> str:
    return "_pc_all_blocks"

def _pc_hemi_key() -> str:
    return "_pc_hemi"

def _ensure_seen_mask(block: str, hemi: str, n_meps: int) -> list[int]:
    key = _pc_seen_mask_key(block, hemi)

    if key not in st.session_state:
        st.session_state[key] = [0] * int(n_meps)

    mask = st.session_state[key]

    if not isinstance(mask, list):
        mask = [0] * int(n_meps)
        st.session_state[key] = mask

    if len(mask) != int(n_meps):
        mask = (mask + [0] * int(n_meps))[: int(n_meps)]
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


def _seen_count_for_block(block: str, hemi: str, n_meps: int) -> tuple[int, int]:
    mask = _ensure_seen_mask(block, hemi, n_meps)
    return (sum(mask), int(n_meps))


def _all_seen_for_block(block: str, hemi: str, n_meps: int) -> bool:
    mask = _ensure_seen_mask(block, hemi, n_meps)
    return all(v == 1 for v in mask)


def _get_cached_n_meps(block: str, hemi: str) -> int | None:
    cache = st.session_state.get(_pc_nmeps_cache_key(hemi))
    if not isinstance(cache, dict):
        return None
    val = cache.get(str(block))
    return int(val) if isinstance(val, int) else None


def _autosave_prev_if_nav_changed(session_file: Path) -> None:
    """
    If user changed block/page/rows/cols since last rerun, save the previous block's
    working peaks_flag list from st.session_state into disk via save_peaks_flag_list().
    """
    prev = st.session_state.get("_pc_prev")
    cur = st.session_state.get("_pc_cur")

    if prev is None or cur is None:
        return

    if prev == cur:
        return

    prev_block = prev["block"]
    prev_work_key = f"peaks_flag_work::{prev_block}::left"

    if prev_work_key not in st.session_state:
        return
    
    # If block changed, clear png cache for the previous block
    if prev["block"] != cur["block"]:
        _clear_png_cache(block=prev_block, hemi="left")


    save_peaks_flag_list(
        session_file=session_file,
        block=prev_block,
        flags=st.session_state[prev_work_key],
        hemi="left",
    )
    st.toast("Auto-saved peaks_flag", icon="💾")

def plot_fig_and_checkbox(
    t_ms: np.ndarray,
    y_data: np.ndarray,
    mep_idx: int,
    updated: list,
    epoch: Epoch,
    block: str,
    mep_min: tuple | None = None,   # (min_ms, min_val)
    mep_max: tuple | None = None,   # (max_ms, max_val)
) -> None:
    mep_min_f = _freeze_extrema(mep_min)
    mep_max_f = _freeze_extrema(mep_max)

    _1, _2 = st.columns([1, 1.6])
    with _2:
        checked = st.checkbox(
            f"MEP {mep_idx}",
            value=bool(updated[mep_idx]),
            key=f"chk_peak::{block}::{mep_idx}",
        )
    updated[mep_idx] = 1 if checked else 0

    # -------- PNG cache (avoid re-rendering matplotlib on reruns) --------
    cache = _get_png_cache(block=block, hemi="left")
    k = _png_key(mep_idx, mep_min, mep_max)


    if k not in cache:
        fig, ax = plt.subplots()

        # stored extrema markers (if available)
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
        plt.close(fig)  # IMPORTANT: don’t leak figures
        cache[k] = buf.getvalue()

    st.image(cache[k], use_container_width=True)



def run_step(meta: dict):
    meta = ensure_metadata()
    meta = ensure_template_loaded(meta)
    session_file = Path(ensure_session_file(meta))

    def _on_next() -> bool:
        all_blocks = st.session_state.get(_pc_all_blocks_key(), [])
        hemi = st.session_state.get(_pc_hemi_key(), "left")

        if not all_blocks:
            st.error("Internal error: no MEP blocks found in state.")
            return False

        not_visited = []
        incomplete = []

        for b in all_blocks:
            n_b = _get_cached_n_meps(b, hemi)

            if n_b is None:
                not_visited.append(b)
                continue

            if n_b <= 0:
                continue

            if not _all_seen_for_block(b, hemi, n_b):
                seen_b, total_b = _seen_count_for_block(b, hemi, n_b)
                incomplete.append((b, seen_b, total_b))

        if not_visited or incomplete:
            msg = "You must plot all MEPs at least once in **all MEP blocks** before advancing.\n\n"

            if not_visited:
                msg += "Blocks not visited yet (open them once so I can count MEPs):\n"
                msg += "\n".join([f"- {b}" for b in not_visited]) + "\n\n"

            if incomplete:
                msg += "Blocks still incomplete:\n"
                msg += "\n".join([f"- {b}: {seen}/{total}" for b, seen, total in incomplete])

            st.toast(msg)
            return False

        return True


    step_nav(
        THIS_STEP,
        step_title = "Peak Checking",
        back_step=PREV_STEP,
        next_step=NEXT_STEP,
        on_next=_on_next,
        disabled_next=False,
    )
    

    # Read session + epoch
    session = read_json(session_file)
    epoch = get_epoch_from_session(session)

    # Only iterate actual MEP blocks
    blocks = [b for b in (meta.get("exp_structure", []) or []) if str(b).lower().endswith("meps")]
    st.session_state[_pc_all_blocks_key()] = [str(b) for b in blocks]
    st.session_state[_pc_hemi_key()] = "left"

    if not blocks:
        st.error("No '*meps' blocks found in meta['exp_structure'].")
        return

    # -------- Controls (keys required) --------
    topA, topB, topC, topD = st.columns([1, 1, 1, 1])
    with topA:
        block = st.selectbox("Block", blocks, index=0, key="_pc_block_sel")
    with topB:
        n_rows = st.number_input("Rows", min_value=1, max_value=12, value=4, step=1, key="_pc_rows_sel")
    with topC:
        n_cols = st.number_input("Cols", min_value=1, max_value=12, value=10, step=1, key="_pc_cols_sel")

    page_size = int(n_rows) * int(n_cols)

    # Load waveforms for this block (needed to know n_meps)
    t_ms, meps = load_meps_for_block(meta, session_file, block_name=block, hemi="left", epoch=epoch, detrend = False)
    if meps.size == 0 or meps.ndim != 2:
        st.warning("No MEPs found for this block (missing pulses or extraction failed).")
        return

    n_meps = int(meps.shape[0])
    n_pages = int(ceil(n_meps / max(1, page_size)))
    _cache_n_meps(block=str(block), hemi="left", n_meps=n_meps)


    with topD:
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=max(1, n_pages),
            value=1,
            step=1,
            key="_pc_page_sel",
        )

    # -------- Autosave tracking --------
    # shift current -> prev
    st.session_state["_pc_prev"] = st.session_state.get("_pc_cur")

    # set current snapshot
    st.session_state["_pc_cur"] = {
        "block": str(block),
        "page": int(page),
        "rows": int(n_rows),
        "cols": int(n_cols),
    }

    _autosave_prev_if_nav_changed(session_file)

    # -------- Working peaks_flag list (per block) --------
    work_key = f"peaks_flag_work::{block}::left"
    if work_key not in st.session_state:
        flags = get_peaks_flag_list(session, block, hemi="left")
        if len(flags) != n_meps:
            flags = (flags + [0] * n_meps)[:n_meps]
        st.session_state[work_key] = flags

    updated = st.session_state[work_key]  # persistent, live list

    # stored extrema (now tuples): [(ms, val), ...]
    session = read_json(session_file)
    left_payload = (((session.get("meps") or {}).get(str(block)) or {}).get("left") or {})
    mins = left_payload.get("min", [])
    maxs = left_payload.get("max", [])


    # -------- Paging --------
    start = (int(page) - 1) * page_size
    end = min(n_meps, start + page_size)
    mep_indices = list(range(start, end))
    st.caption(f"Showing MEPs **{start}–{end-1}** out of **{n_meps}** (0-indexed).")
    seen, total = _seen_count_for_block(str(block), "left", n_meps)
    st.caption(f"Seen in this block: **{seen}/{total}**")
    _mark_meps_seen(block=str(block), hemi="left", n_meps=n_meps, mep_indices=mep_indices)


    # -------- Grid --------
    for i, mep_idx in enumerate(mep_indices):
        c = i % int(n_cols)
        if c == 0:
            columns = st.columns(int(n_cols))

        with columns[c]:
            mep_min = mins[mep_idx] if mep_idx < len(mins) else None
            mep_max = maxs[mep_idx] if mep_idx < len(maxs) else None

            plot_fig_and_checkbox(
                t_ms=t_ms,
                y_data=meps[mep_idx],
                mep_idx=mep_idx,
                updated=updated,
                epoch=epoch,
                block=str(block),
                mep_min=mep_min,
                mep_max=mep_max,
            )
            
    # # Optional safety net
    # colA, colB = st.columns([1, 1])
    # with colA:
    #     if st.button("Save now", type="primary"):
    #         save_peaks_flag_list(session_file=session_file, block=str(block), flags=updated, hemi="left")
    #         st.success("Saved peaks_flag.")
    # with colB:
    #     st.caption("Auto-saves when you change block/page/rows/cols.")

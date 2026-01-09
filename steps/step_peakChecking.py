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
    "figure.dpi": 200,
})

PREV_STEP = "mep_window"
THIS_STEP = "peak_checking"


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
    """
    Plot one MEP + checkbox.

    If provided, plot stored extrema markers:
      mep_min = (min_ms, min_val)
      mep_max = (max_ms, max_val)
    """
    _1, _2, = st.columns([1,1.6])
    with _2:
        checked = st.checkbox(
            f"MEP {mep_idx}",
            value=bool(updated[mep_idx]),
            key=f"chk_peak::{block}::{mep_idx}",
        )
    updated[mep_idx] = 1 if checked else 0

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
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, pos: f"{v:8.0f}")
    )
        
    # ax.set_title(f"MEP {mep_idx}", )
    ax.set_xticks([])
    ax.margins(0)
    # fig.tight_layout(pad=0)
    # ax.set_position([0.22, 0.12, 0.78, 0.78])
    st.pyplot(fig, clear_figure=True)



def run_step(meta: dict):
    meta = ensure_metadata()
    meta = ensure_template_loaded(meta)
    session_file = Path(ensure_session_file(meta))

    step_nav(THIS_STEP, back_step=PREV_STEP, disabled_next=True)

    render_text(
        "Peak checking",
        font_color="black",
        font_weight="normal",
        horizontal_alignment="center",
        heading_level=1,
        nowrap=True,
    )

    # Read session + epoch
    session = read_json(session_file)
    epoch = get_epoch_from_session(session)

    # Only iterate actual MEP blocks
    blocks = [b for b in (meta.get("exp_structure", []) or []) if str(b).lower().endswith("meps")]
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
        n_cols = st.number_input("Cols", min_value=1, max_value=12, value=5, step=1, key="_pc_cols_sel")

    page_size = int(n_rows) * int(n_cols)

    # Load waveforms for this block (needed to know n_meps)
    t_ms, meps = load_meps_for_block(meta, session_file, block_name=block, hemi="left", epoch=epoch, detrend = False)
    if meps.size == 0 or meps.ndim != 2:
        st.warning("No MEPs found for this block (missing pulses or extraction failed).")
        return

    n_meps = int(meps.shape[0])
    n_pages = int(ceil(n_meps / max(1, page_size)))

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

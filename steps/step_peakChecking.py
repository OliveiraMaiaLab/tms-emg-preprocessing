"""
steps/step_peakChecking.py
--------------------------
Step 5: Peak checking

- Select a block
- Plot one axis per MEP (grid)
- Mark MEP window bounds (vertical lines)
- Mark min/max within that window (markers)
- Checkbox per MEP to set peaks_flag (0/1)
- Save peaks_flag into session["meps"][block]["left"]["peaks_flag"]
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from utils.persistence import (
    ensure_metadata,
    ensure_template_loaded,
    ensure_session_file,
)
from utils.layout import render_text, step_nav

from utils.peak_checking_io import (
    Epoch,
    read_json,
    get_epoch_from_session,
    get_peaks_flag_list,
    save_peaks_flag_list,
)
from utils.mep_loading import load_meps_for_block


PREV_STEP = "mep_window"
THIS_STEP = "peak_checking"


def _epoch_mask(t_ms: np.ndarray, epoch: Epoch) -> np.ndarray:
    return (t_ms >= epoch.tmin_ms) & (t_ms <= epoch.tmax_ms)


def _plot_mep_grid(
    t_ms: np.ndarray,
    meps: np.ndarray,
    mep_indices: List[int],
    epoch: Epoch,
    ncols: int = 4,
) -> plt.Figure:
    n = len(mep_indices)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 2.4 * nrows),
    )
    axes = np.atleast_1d(axes).ravel()
    m = _epoch_mask(t_ms, epoch)

    for ax_i, ax in enumerate(axes):
        if ax_i >= n:
            ax.axis("off")
            continue

        mep_idx = mep_indices[ax_i]
        y = meps[mep_idx, :]

        ax.plot(t_ms, y)
        ax.axvline(epoch.tmin_ms, linestyle="--")
        ax.axvline(epoch.tmax_ms, linestyle="--")

        if np.any(m):
            y_epoch = y[m]
            t_epoch = t_ms[m]
            i_min = int(np.argmin(y_epoch))
            i_max = int(np.argmax(y_epoch))
            ax.scatter([t_epoch[i_min]], [y_epoch[i_min]], marker="o")
            ax.scatter([t_epoch[i_max]], [y_epoch[i_max]], marker="o")

        ax.set_title(f"MEP {mep_idx}", fontsize=10)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("uV")

    fig.tight_layout()
    return fig


def run_step(meta: dict):
    meta = ensure_metadata()
    meta = ensure_template_loaded(meta)
    session_file = Path(ensure_session_file(meta))

    step_nav(
        THIS_STEP,
        back_step=PREV_STEP,
        disabled_next=True,
    )

    render_text(
        "Peak checking",
        font_color="black",
        font_weight="normal",
        horizontal_alignment="center",
        heading_level=1,
        nowrap=True,
    )

    session = read_json(session_file)
    epoch = get_epoch_from_session(session)

    # Only iterate actual MEP blocks
    blocks = [b for b in (meta.get("exp_structure", []) or []) if str(b).lower().endswith("meps")]
    if not blocks:
        st.error("No '*meps' blocks found in meta['exp_structure'].")
        return

    topA, topB, topC = st.columns([1.3, 1.3, 1.6])
    with topA:
        block = st.selectbox("Block", blocks, index=0)
    with topB:
        page_size = st.slider("MEPs per page", min_value=5, max_value=40, value=20, step=5)
    with topC:
        st.write("")
        st.write(f"MEP window: **{epoch.tmin_ms:.1f} → {epoch.tmax_ms:.1f} ms**")

    # Load waveforms for this block
    t_ms, meps = load_meps_for_block(meta, session_file, block_name=block, hemi="left")
    if meps.size == 0 or meps.ndim != 2:
        st.warning("No MEPs found for this block (missing pulses or extraction failed).")
        return

    n_meps = int(meps.shape[0])

    # existing flags (0/1 list)
    peaks_flag = get_peaks_flag_list(session, block, hemi="left")
    if len(peaks_flag) != n_meps:
        peaks_flag = (peaks_flag + [0] * n_meps)[:n_meps]

    # paging
    n_pages = int(np.ceil(n_meps / page_size))
    page = st.number_input("Page", min_value=1, max_value=max(1, n_pages), value=1, step=1)

    start = (page - 1) * page_size
    end = min(n_meps, start + page_size)
    mep_indices = list(range(start, end))
    st.caption(f"Showing MEPs **{start}–{end-1}** out of **{n_meps}** (0-indexed).")

    # checkboxes
    st.markdown("**Flag MEPs (peaks_flag, saved per block):**")
    cb_cols = st.columns(5)
    updated = peaks_flag[:]
    for j, mep_idx in enumerate(mep_indices):
        with cb_cols[j % 5]:
            checked = st.checkbox(
                f"Flag {mep_idx}",
                value=(updated[mep_idx] == 1),
                key=f"peaksflag__{block}__{mep_idx}",
            )
            updated[mep_idx] = 1 if checked else 0

    # plot
    fig = _plot_mep_grid(
        t_ms=t_ms,
        meps=meps,
        mep_indices=mep_indices,
        epoch=epoch,
        ncols=4,
    )
    st.pyplot(fig, clear_figure=True)

    # save
    s1, s2 = st.columns([1, 2])
    with s1:
        if st.button("Save flags"):
            try:
                save_peaks_flag_list(session_file, block=block, flags=updated, hemi="left")
                st.toast("Saved peaks_flag ✅")
            except Exception as e:
                st.toast(f"Error saving peaks_flag: {e}", icon="❌")

    with s2:
        st.write("Flags are stored as 0/1 in the session JSON under meps[block][hemi]['peaks_flag'].")

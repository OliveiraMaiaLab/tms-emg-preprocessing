# app/bk_embedding/mepOverlap.py
# -*- coding: utf-8 -*-
"""
app/bk_embedding/mepOverlap.py
------------------------------
Embedded Bokeh server (single plot only).

- Reads pulses from session_file["meps"][block]["left"]["pulses"] (sample indices)
- Extracts [-0.1s, +0.4s] epochs from raw EMG and plots them aligned to pulse at x=0.
- RangeTool highlights a selectable window without moving the plot.
"""

from __future__ import annotations

import time
import threading
import json
from typing import Callable

import numpy as np

from bokeh.server.server import Server
from bokeh.plotting import figure
from bokeh.layouts import column as bk_column
from bokeh.models import (
    ColumnDataSource,
    RangeTool,
    Range1d,
    Span,
    WheelZoomTool,
    Div,
)

from app.utils.tms_module import load_data


def _read_pulses(
    session_file: str,
    exp_structure: list[str] | None,
    hemi: str = "left",
) -> list[int]:
    """
    Collect pulses across all blocks that end with "meps" (case-insensitive),
    falling back to all blocks present in session JSON if exp_structure doesn't contain any.

    Expected schema:
      session["meps"][block][hemi]["pulses"] = [sample_idx, ...]
    """
    try:
        with open(session_file, "r") as f:
            js = json.load(f)
    except Exception:
        return []

    meps = js.get("meps") or {}
    if not isinstance(meps, dict):
        return []

    blocks = [b for b in (exp_structure or []) if str(b).lower().endswith("meps")]
    if not blocks:
        blocks = list(meps.keys())

    out: list[int] = []
    for b in blocks:
        hemi_payload = (meps.get(b) or {}).get(hemi, {})
        if not isinstance(hemi_payload, dict):
            continue

        pulses = hemi_payload.get("pulses", [])
        if not isinstance(pulses, list):
            continue

        for p in pulses:
            try:
                out.append(int(p))
            except Exception:
                pass

    return sorted(set(out))


def _extract_epochs_from_pulses(
    data: np.ndarray,
    fs: float,
    pulses: list[int],
    pre_s: float = 0.100,
    post_s: float = 0.400,
    channel_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      t_sec: (n_win,) time vector centered at pulse (0 at pulse)
      trials: (n_trials, n_win) epochs
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("Expected data shape (n_channels, n_samples).")

    n_ch, n_samp = data.shape
    if not (0 <= channel_index < n_ch):
        raise ValueError(f"channel_index={channel_index} out of range for {n_ch} channels.")

    sig = data[channel_index]
    pre_n = int(round(pre_s * fs))
    post_n = int(round(post_s * fs))
    n_win = pre_n + post_n

    t_sec = (np.arange(n_win) - pre_n) / fs

    trials = []
    for pulse_i in pulses:
        a = int(pulse_i) - pre_n
        b = int(pulse_i) + post_n
        if a < 0 or b > n_samp:
            continue
        trials.append(sig[a:b])

    if not trials:
        raise RuntimeError("No epochs extracted (empty/clipped pulses).")

    trials = np.asarray(trials, dtype=float)

    # baseline subtract each trial using pre-pulse window
    trials -= trials[:, :pre_n].mean(axis=1, keepdims=True)
    return t_sec, trials


def _make_plot(
    t_sec: np.ndarray,
    trials: np.ndarray,
    range_sink: Callable[[float, float], None] | None = None,
):
    """
    Build overlay plot + mean, with a RangeTool selection.
    range_sink(start_s, end_s) receives selection in seconds.
    """
    plot_xlims_ms = [-10, 60]
    default_sel_ms = [15, 35]

    trials_mv = trials / 1000.0
    t_ms = t_sec * 1000.0
    mean = trials_mv.mean(axis=0)

    y_min, y_max = float(trials_mv.min()), float(trials_mv.max())

    p = figure(
        height=700,
        title="MEP epochs (overlay + mean)",
        tools="reset,box_zoom",
        x_range=Range1d(start=plot_xlims_ms[0], end=plot_xlims_ms[1]),
        y_range=Range1d(start=y_min, end=y_max),
        x_axis_label="Time (ms)",
        y_axis_label="Amplitude (mV)",
        sizing_mode="stretch_both",
    )
    p.title.align = "center"

    # TMS pulse marker
    p.add_layout(
        Span(
            location=0.0,
            dimension="height",
            line_color="black",
            line_width=1,
            line_alpha=0.5,
            line_dash="dashed",
        )
    )
    # legend entry hack (Bokeh doesn't legend Spans)
    p.line(
        [np.nan, np.nan],
        [np.nan, np.nan],
        line_color="black",
        line_width=1,
        line_alpha=0.5,
        line_dash="dashed",
        legend_label="TMS pulse",
    )

    # overlays
    for k in range(trials_mv.shape[0]):
        src = ColumnDataSource(dict(x=t_ms, y=trials_mv[k]))
        p.line("x", "y", source=src, line_width=1, line_alpha=0.12, line_color="gray")

    # mean
    mean_src = ColumnDataSource(dict(x=t_ms, y=mean))
    p.line("x", "y", source=mean_src, line_width=2, line_color="black", legend_label="mean MEP")

    sel = Range1d(start=default_sel_ms[0], end=default_sel_ms[1])
    range_tool = RangeTool(x_range=sel)
    range_tool.overlay.fill_color = "gray"
    range_tool.overlay.fill_alpha = 0.25

    y_zoom = WheelZoomTool(dimensions="height")
    p.add_tools(range_tool, y_zoom)
    p.toolbar.active_scroll = y_zoom

    range_div = Div(
        text=f"<b>Selected range:</b> {default_sel_ms[0]:.1f} – {default_sel_ms[1]:.1f} ms",
        style={"text-align": "center"},
        sizing_mode="stretch_width",
    )

    def _on_sel(attr, old, new):
        s_ms = float(sel.start)
        e_ms = float(sel.end)
        range_div.text = f"<b>Selected range:</b> {s_ms:.1f} – {e_ms:.1f} ms"
        if range_sink:
            range_sink(s_ms / 1000.0, e_ms / 1000.0)

    sel.on_change("start", _on_sel)
    sel.on_change("end", _on_sel)
    _on_sel(None, None, None)

    return bk_column(range_div, p, sizing_mode="stretch_both")


def start_bokeh_app(meta, session_file: str, exp_structure, SCRIPT_DIR, ranges_store, ranges_lock):
    """
    Launch a Bokeh app in a background thread; return the chosen port.

    ranges_store is updated via sink:
      ranges_store["epoch_window"] = (start_s, end_s)
    """

    def bkapp(doc):
        fs = float(meta["sampling_rate"])
        pulses = _read_pulses(session_file, exp_structure, hemi="left")
        if not pulses:
            doc.add_root(Div(text="<b>No MEP pulses found in session file.</b>", style={"color": "crimson"}))
            return

        data, _tms = load_data(meta["input_file"], channels=meta.get("channels"))

        try:
            t_sec, trials = _extract_epochs_from_pulses(
                data=data,
                fs=fs,
                pulses=pulses,
                pre_s=0.100,
                post_s=0.400,
                channel_index=0,  # left = 0 in tms_module.load_data
            )
        except Exception as e:
            doc.add_root(Div(text=f"<b>Failed to extract epochs:</b> {e}", style={"color": "crimson"}))
            return

        def sink(start_s: float, end_s: float):
            with ranges_lock:
                ranges_store["epoch_window"] = (start_s, end_s)

        doc.add_root(_make_plot(t_sec, trials, range_sink=sink))

    port_holder: list[int] = []

    def run_server(holder: list[int]):
        server = Server(
            {"/bkapp": bkapp},
            port=0,
            allow_websocket_origin=["*"],
            address="127.0.0.1",
            use_xheaders=True,
        )
        server.start()
        holder.append(server.port)
        server.io_loop.start()

    t = threading.Thread(target=run_server, args=(port_holder,), daemon=True)
    t.start()

    while not port_holder:
        time.sleep(0.1)

    return port_holder[0]

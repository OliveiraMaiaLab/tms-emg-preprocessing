# app/bk_embedding/mepOverlap.py
# -*- coding: utf-8 -*-
"""
app/bk_embedding/mepOverlap.py
------------------------------
Embedded Bokeh server — MEP epoch overlay + RangeTool window selector.

If the session already has a mep_window, it is used as the initial selection.
"""

from __future__ import annotations

import time
import threading
import json
from pathlib import Path
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

from app.utils.tms_module import _load_data_cached


def _read_pulses(
    session_file: str,
    exp_structure: list[str] | None,
    hemi: str = "left",
) -> list[int]:
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


def _read_initial_window_ms(session_file: str) -> list[float] | None:
    """
    Read mep_window from session JSON and convert to milliseconds.
    Returns [start_ms, end_ms] if a valid window is stored, else None.
    """
    try:
        js = json.loads(Path(session_file).read_text())
        w  = js.get("mep_window", [None, None])
        if isinstance(w, list) and len(w) == 2 and w[0] is not None and w[1] is not None:
            start_ms = float(w[0]) * 1000.0
            end_ms   = float(w[1]) * 1000.0
            if start_ms < end_ms:
                return [start_ms, end_ms]
    except Exception:
        pass
    return None


def _extract_epochs_from_pulses(
    data: np.ndarray,
    fs: float,
    pulses: list[int],
    pre_s: float = 0.100,
    post_s: float = 0.400,
    channel_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    data  = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("Expected data shape (n_channels, n_samples).")

    n_ch, n_samp = data.shape
    if not (0 <= channel_index < n_ch):
        raise ValueError(f"channel_index={channel_index} out of range for {n_ch} channels.")

    sig   = data[channel_index]
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
    trials -= trials[:, :pre_n].mean(axis=1, keepdims=True)
    return t_sec, trials


def _make_plot(
    t_sec: np.ndarray,
    trials: np.ndarray,
    range_sink: Callable[[float, float], None] | None = None,
    initial_window_ms: list[float] | None = None,
):
    """
    Build overlay plot + mean with a RangeTool window selector.

    If initial_window_ms is provided (from an existing session mep_window),
    the RangeTool starts at that selection instead of the hard-coded default.
    """
    plot_xlims_ms = [-10, 60]
    # Use the stored window if available, otherwise fall back to the default.
    sel_ms = initial_window_ms if initial_window_ms is not None else [15.0, 35.0]

    trials_mv = trials / 1000.0
    t_ms      = t_sec * 1000.0
    mean      = trials_mv.mean(axis=0)

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

    p.add_layout(
        Span(location=0.0, dimension="height",
             line_color="black", line_width=1, line_alpha=0.5, line_dash="dashed")
    )
    p.line([np.nan, np.nan], [np.nan, np.nan],
           line_color="black", line_width=1, line_alpha=0.5,
           line_dash="dashed", legend_label="TMS pulse")

    for k in range(trials_mv.shape[0]):
        src = ColumnDataSource(dict(x=t_ms, y=trials_mv[k]))
        p.line("x", "y", source=src, line_width=1, line_alpha=0.12, line_color="gray")

    mean_src = ColumnDataSource(dict(x=t_ms, y=mean))
    p.line("x", "y", source=mean_src, line_width=2, line_color="black", legend_label="mean MEP")

    sel        = Range1d(start=sel_ms[0], end=sel_ms[1])
    range_tool = RangeTool(x_range=sel)
    range_tool.overlay.fill_color = "gray"
    range_tool.overlay.fill_alpha = 0.25

    y_zoom = WheelZoomTool(dimensions="height")
    p.add_tools(range_tool, y_zoom)
    p.toolbar.active_scroll = y_zoom

    range_div = Div(
        text=f"<b>Selected range:</b> {sel_ms[0]:.1f} – {sel_ms[1]:.1f} ms",
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
    sel.on_change("end",   _on_sel)
    _on_sel(None, None, None)  # initialise sink with current selection

    return bk_column(range_div, p, sizing_mode="stretch_both")


def start_bokeh_app(meta, session_file: str, exp_structure, SCRIPT_DIR, ranges_store, ranges_lock):
    def bkapp(doc):
        fs             = float(meta["sampling_rate"])
        hemi           = (meta.get("hemispheres") or ["left"])[0]
        channel_index  = {"left": 0, "right": 1}.get(str(hemi).lower(), 0)

        pulses = _read_pulses(session_file, exp_structure, hemi=hemi)
        if not pulses:
            doc.add_root(Div(text="<b>No MEP pulses found in session file.</b>",
                             style={"color": "crimson"}))
            return

        channels       = meta.get("channels") or [3, 1, 2]
        channels_tuple = tuple(int(c) for c in channels)
        data, _tms    = _load_data_cached(str(meta["input_file"]), channels_tuple, int(fs))

        try:
            t_sec, trials = _extract_epochs_from_pulses(
                data=data, fs=fs, pulses=pulses,
                pre_s=0.100, post_s=0.400,
                channel_index=channel_index,
            )
        except Exception as e:
            doc.add_root(Div(text=f"<b>Failed to extract epochs:</b> {e}",
                             style={"color": "crimson"}))
            return

        # Pre-fill the RangeTool from any existing mep_window in the session.
        initial_window_ms = _read_initial_window_ms(session_file)

        def sink(start_s: float, end_s: float):
            with ranges_lock:
                ranges_store["epoch_window"] = (start_s, end_s)

        doc.add_root(_make_plot(t_sec, trials, range_sink=sink,
                                initial_window_ms=initial_window_ms))

    port_holder: list[int] = []

    def run_server(holder: list[int]):
        server = Server(
            {"/bkapp": bkapp}, port=0,
            allow_websocket_origin=["*"], address="127.0.0.1", use_xheaders=True,
        )
        server.start()
        holder.append(server.port)
        server.io_loop.start()

    t = threading.Thread(target=run_server, args=(port_holder,), daemon=True)
    t.start()

    for _ in range(50):
        if port_holder:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("mepOverlap Bokeh server failed to start within 5 seconds.")

    return port_holder[0]

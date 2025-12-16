"""
bk_mepOverlap_embedding.py
--------------------------
Embedded Bokeh server (single plot only).

- Reads epoch windows from session_file["meps"][block]["left"]
  where each epoch is [beg_s, end_s] in absolute seconds
  (beg = pulse - 0.1, end = pulse + 0.4).
- Extracts those epochs from raw EMG and plots them aligned to pulse at x=0.
- RangeTool highlights a selectable window without moving the plot.
"""

import time
import threading
import json
import numpy as np

from bokeh.server.server import Server
from bokeh.plotting import figure
from bokeh.layouts import column as bk_column, row as bk_row
from bokeh.models import (
    ColumnDataSource,
    RangeTool,
    Range1d,
    Span,
    WheelZoomTool,
    Div,
    Spacer,
)

from utils.tms_module import load_data


def _read_left_epoch_windows(session_file: str, exp_structure: list[str] | None) -> list[list[float]]:
    """
    Collect epoch windows from session JSON:
        meps.<block>.left = [[beg,end], ...]
    Returns a flat list of [beg_s, end_s] pairs.
    """
    with open(session_file, "r") as f:
        js = json.load(f)

    meps = js.get("meps") or {}
    if not isinstance(meps, dict):
        return []

    # Prefer blocks from exp_structure that end with "meps"
    blocks = [b for b in (exp_structure or []) if str(b).endswith("meps")]
    if not blocks:
        blocks = list(meps.keys())

    out = []
    for b in blocks:
        left = (meps.get(b) or {}).get("left", [])
        if not isinstance(left, list):
            continue
        for pair in left:
            if not (isinstance(pair, (list, tuple)) and len(pair) >= 2):
                continue
            try:
                beg = float(pair[0])
                end = float(pair[1])
            except Exception:
                continue
            if beg < end:
                out.append([beg, end])

    out.sort(key=lambda x: x[0])
    return out


def _extract_epochs_from_windows(
    data: np.ndarray,
    fs: float,
    windows_s: list[list[float]],
    pre_s: float = 0.100,
    post_s: float = 0.400,
    channel_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract epochs aligned to pulse (x=0).

    We assume windows were created as [pulse-pre, pulse+post],
    so pulse_time = beg + pre.

    Returns:
      t_sec  : (n_samples,) relative time around pulse in seconds
      trials : (n_trials, n_samples)
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
    for beg_s, _end_s in windows_s:
        pulse_s = float(beg_s) + pre_s
        pulse_i = int(round(pulse_s * fs))

        a = pulse_i - pre_n
        b = pulse_i + post_n
        if a < 0 or b > n_samp:
            continue

        trials.append(sig[a:b])

    if not trials:
        raise RuntimeError("No epochs extracted (empty/clipped windows).")

    trials = np.asarray(trials, dtype=float)

    # baseline correct using pre-stim window
    trials -= trials[:, :pre_n].mean(axis=1, keepdims=True)

    return t_sec, trials


def _make_plot(t_sec: np.ndarray, trials: np.ndarray, range_sink=None):
    """
    Plot many trials (faded) + mean (solid). X axis in ms.
    RangeTool updates range_sink(start_s, end_s) in seconds.
    """
    # You can change these to whatever you want to display/select
    plot_xlims_ms = [-10, 60]
    default_sel_ms = [15, 35]

    trials_mv = trials/1000

    t_ms = t_sec * 1000.0
    mean = trials_mv.mean(axis=0)

    y_min, y_max = float(trials_mv.min()), float(trials_mv.max())

    p = figure(
        height=700,                 # initial; will be overridden by stretch_both sizing
        title="MEP epochs (overlay + mean)",
        tools="reset,box_zoom",
        x_range=Range1d(start=plot_xlims_ms[0], end=plot_xlims_ms[1]),
        y_range=Range1d(start=y_min, end=y_max),
        x_axis_label="Time (ms)",
        y_axis_label="Amplitude (mV)",
        sizing_mode="stretch_both", # <-- important
    )
    p.title.align = "center"
    p.add_layout(Span(location=0.0, dimension="height", line_color="black", line_width=1,
                              line_alpha=0.5, line_dash='dashed'))

    # faded overlays
    for k in range(trials_mv.shape[0]):
        src = ColumnDataSource(dict(x=t_ms, y=trials_mv[k]))
        p.line("x", "y", source=src, line_width=1, line_alpha=0.12, line_color= "gray")

    # mean
    mean_src = ColumnDataSource(dict(x=t_ms, y=mean))
    p.line("x", "y", source=mean_src, line_width=2, line_color= "black")

    # RangeTool selection range (in ms on this axis)
    sel = Range1d(start=default_sel_ms[0], end=default_sel_ms[1])
    range_tool = RangeTool(x_range=sel)
    range_tool.overlay.fill_color = "gray"
    range_tool.overlay.fill_alpha = 0.25

    y_zoom = WheelZoomTool(dimensions="height")
    p.add_tools(range_tool, y_zoom)
    p.toolbar.active_scroll = y_zoom

    # selection label
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
    Starts the Bokeh app and returns the port.
    Updates ranges_store["epoch_window"] = (start_s, end_s).
    """
    def bkapp(doc):
        fs = float(meta["sampling_rate"])

        # 1) read epoch windows (absolute seconds)
        windows_s = _read_left_epoch_windows(session_file, exp_structure)
        if not windows_s:
            doc.add_root(Div(text="<b>No MEP epochs found in session file.</b>", style={"color": "crimson"}))
            return

        # 2) load EMG + extract epochs
        data, _ = load_data(meta["input_file"])
        try:
            t_sec, trials = _extract_epochs_from_windows(
                data=data,
                fs=fs,
                windows_s=windows_s,
                pre_s=0.100,
                post_s=0.400,
                channel_index=0,  # left channel = 0 (change if needed)
            )
        except Exception as e:
            doc.add_root(Div(text=f"<b>Failed to extract epochs:</b> {e}", style={"color": "crimson"}))
            return

        # 3) plot + selection sink
        def sink(start_s, end_s):
            with ranges_lock:
                ranges_store["epoch_window"] = (start_s, end_s)

        doc.add_root(_make_plot(t_sec, trials, range_sink=sink))

    port_holder = []

    def run_server(holder):
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

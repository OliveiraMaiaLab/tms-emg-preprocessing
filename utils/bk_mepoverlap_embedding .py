"""
bk_segmentation_embedding.py
---------------------------
Embedded Bokeh server (single plot only).

Test version:
- Overlaid noisy sinusoids + mean
- RangeTool on the same plot
"""

import time
import threading
import numpy as np

from bokeh.server.server import Server
from bokeh.plotting import figure
from bokeh.layouts import column as bk_column
from bokeh.models import (
    ColumnDataSource,
    RangeTool,
    Span,
    Range1d,
    WheelZoomTool,
)


# -------------------- test data generator --------------------

def _get_sinusoid_trials(fs: float, n_trials: int = 60, freq: float = 20.0):
    """
    Generate sinusoidal test trials in [-50ms, +150ms].

    Returns
    -------
    t : (n_samples,) time vector in seconds
    trials : (n_trials, n_samples)
    """
    t = np.arange(int(round(0.200 * fs))) / fs - 0.050

    trials = np.zeros((n_trials, t.size))
    for k in range(n_trials):
        amp   = 1.0 + 0.15 * np.random.randn()
        phase = 0.15 * np.random.randn()
        noise = 0.10 * np.random.randn(t.size)
        trials[k] = amp * np.sin(2 * np.pi * freq * t + phase) + noise

    return t, trials


# -------------------- plotting --------------------

def _epoch_overlay_plot(t, trials, range_sink=None):
    mean = trials.mean(axis=0)
    y_min, y_max = float(np.min(trials)), float(np.max(trials))

    p = figure(
        height=420,
        title="Sinusoid overlays + mean (test)",
        tools="xpan,xwheel_zoom,reset,box_zoom",
        active_scroll="xwheel_zoom",
        x_range=Range1d(start=-0.050, end=0.150),
        y_range=Range1d(start=y_min, end=y_max),
        x_axis_label="Time (s)",
        y_axis_label="Amplitude",
        sizing_mode="stretch_width",
    )

    # x = 0 reference
    p.add_layout(Span(location=0.0, dimension="height", line_color="black", line_width=2))

    # faded trials
    for k in range(trials.shape[0]):
        src = ColumnDataSource(dict(x=t, y=trials[k]))
        p.line("x", "y", source=src, line_width=1, line_alpha=0.12)

    # mean
    mean_src = ColumnDataSource(dict(x=t, y=mean))
    p.line("x", "y", source=mean_src, line_width=3)

    # RangeTool on SAME plot
    range_tool = RangeTool(x_range=p.x_range)
    range_tool.overlay.fill_color = "gray"
    range_tool.overlay.fill_alpha = 0.25

    y_zoom = WheelZoomTool(dimensions="height")
    p.add_tools(range_tool, y_zoom)

    if range_sink is not None:
        def _on_xrange(attr, old, new):
            try:
                range_sink(float(p.x_range.start), float(p.x_range.end))
            except Exception as e:
                print(f"[range_sink error] {e}")

        p.x_range.on_change("start", _on_xrange)
        p.x_range.on_change("end", _on_xrange)
        _on_xrange(None, None, None)

    return bk_column(p, sizing_mode="stretch_both")


# -------------------- app --------------------

def start_bokeh_app(meta, session_file: str, exp_structure, SCRIPT_DIR, ranges_store, ranges_lock):
    """
    Launch Bokeh app in background thread.

    Writes:
      ranges_store["epoch_window"] = (start, end)
    """
    def bkapp(doc):
        fs = float(meta.get("sampling_rate", 4000))

        # sinusoid test data
        t, trials = _get_sinusoid_trials(fs=fs, n_trials=80, freq=25.0)

        def sink(start, end):
            with ranges_lock:
                ranges_store["epoch_window"] = (start, end)

        layout = _epoch_overlay_plot(t, trials, range_sink=sink)
        doc.add_root(layout)

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

"""
bk_segmentation_embedding.py
---------------------------
Embedded Bokeh server (single plot only).

Test version:
- Overlaid noisy sinusoids + mean
- Fixed x-range (ms)
- RangeTool highlights a window without moving the plot
- Plot dynamically centered in iframe
"""

import time
import threading
import numpy as np

from bokeh.server.server import Server
from bokeh.plotting import figure
from bokeh.layouts import column as bk_column, row as bk_row
from bokeh.models import (
    ColumnDataSource,
    RangeTool,
    Span,
    Range1d,
    WheelZoomTool,
    Div,
    Spacer,
)


# -------------------- test data generator --------------------

def _get_sinusoid_trials(fs: float, n_trials: int = 60, freq: float = 20.0):
    """
    Generate sinusoidal test trials in [-50ms, +150ms].

    Returns
    -------
    t_sec : (n_samples,) time vector in seconds
    trials : (n_trials, n_samples)
    """
    t_sec = np.arange(int(round(0.200 * fs))) / fs - 0.050

    trials = np.zeros((n_trials, t_sec.size))
    for k in range(n_trials):
        amp   = 1.0 + 0.15 * np.random.randn()
        phase = 0.15 * np.random.randn()
        noise = 0.10 * np.random.randn(t_sec.size)
        trials[k] = amp * np.sin(2 * np.pi * freq * t_sec + phase) + noise

    return t_sec, trials


# -------------------- plotting --------------------

def _epoch_overlay_plot(t_sec, trials, range_sink=None):
    """
    Single centered plot with:
    - faded trial overlays
    - bold mean
    - fixed x-range in ms
    - RangeTool highlighting a selectable window
    """

    # Convert time to ms for plotting
    t_ms = t_sec * 1000.0

    mean = trials.mean(axis=0)
    y_min, y_max = float(np.min(trials)), float(np.max(trials))

    # Fixed plot x-range (ms)
    fixed_x = Range1d(start=-50.0, end=150.0)

    p = figure(
        height=420,
        title="Sinusoid overlays + mean (test)",
        tools="reset,box_zoom",
        x_range=fixed_x,
        y_range=Range1d(start=y_min, end=y_max),
        x_axis_label="Time (ms)",
        y_axis_label="Amplitude",
        sizing_mode="stretch_width",
        max_width=950,      # cap growth
        min_width=420,      # avoid collapse on small iframes
    )

    p.title.align = "center"

    # x = 0 reference line
    p.add_layout(
        Span(location=0.0, dimension="height", line_color="black", line_width=2)
    )

    # Faded trials
    for k in range(trials.shape[0]):
        src = ColumnDataSource(dict(x=t_ms, y=trials[k]))
        p.line("x", "y", source=src, line_width=1, line_alpha=0.12)

    # Mean trace
    mean_src = ColumnDataSource(dict(x=t_ms, y=mean))
    p.line("x", "y", source=mean_src, line_width=3)

    # Selection range (ms)
    selection_range = Range1d(start=15.0, end=50.0)

    range_tool = RangeTool(x_range=selection_range)
    range_tool.overlay.fill_color = "gray"
    range_tool.overlay.fill_alpha = 0.25

    y_zoom = WheelZoomTool(dimensions="height")
    p.add_tools(range_tool, y_zoom)
    p.toolbar.active_scroll = y_zoom

    # Display selected range
    range_div = Div(
        text="<b>Selected range:</b> 15.0 – 50.0 ms",
        style={"text-align": "center"},
        sizing_mode="stretch_width",
    )

    def update_selection(attr, old, new):
        s_ms = float(selection_range.start)
        e_ms = float(selection_range.end)

        range_div.text = (
            f"<b>Selected range:</b> {s_ms:.1f} – {e_ms:.1f} ms"
        )

        if range_sink is not None:
            try:
                # store in SECONDS
                range_sink(s_ms / 1000.0, e_ms / 1000.0)
            except Exception as ex:
                print(f"[range_sink error] {ex}")

    selection_range.on_change("start", update_selection)
    selection_range.on_change("end", update_selection)
    update_selection(None, None, None)

    # Center plot dynamically using spacers
    centered_plot = bk_row(
        Spacer(sizing_mode="stretch_width"),
        p,
        Spacer(sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )

    return bk_column(range_div, centered_plot, sizing_mode="stretch_both")


# -------------------- app --------------------

def start_bokeh_app(meta, session_file: str, exp_structure, SCRIPT_DIR, ranges_store, ranges_lock):
    """
    Launch Bokeh app in background thread.

    Writes:
      ranges_store["epoch_window"] = (start, end)  # seconds
    """
    def bkapp(doc):
        fs = float(meta.get("sampling_rate", 4000))

        # Test sinusoid data
        t_sec, trials = _get_sinusoid_trials(
            fs=fs,
            n_trials=80,
            freq=25.0,
        )

        def sink(start, end):
            with ranges_lock:
                ranges_store["epoch_window"] = (start, end)

        layout = _epoch_overlay_plot(t_sec, trials, range_sink=sink)
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

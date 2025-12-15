"""
bk_segmentation_embedding.py
--------------
Creates and runs the embedded Bokeh server and right-hand controls, using a
FLAT 'segmentation' schema in the session JSON:

"segmentation": {
  "<block>": [[start, end]] | []   # no hemisphere nesting
}

- Shows EMG channels with a RangeTool
- Lets user pick a window and "Set" it per block (convenience)
- Prefills from session JSON if present (as "s - e" per block)
- On Save: parses each text input and writes the flat schema above
"""
import time
import threading
import json
import re
import numpy as np

from bokeh.server.server import Server
from bokeh.plotting import figure
from bokeh.layouts import column as bk_column, row as bk_row
from bokeh.models import (
    ColumnDataSource,
    RangeTool,
    Span,
    Button,
    Div,
    Range1d,
    TextInput,
    CustomJS,
    WheelZoomTool,
)

from utils.persistence import write_segmentation_ranges
from utils.tms_module import load_data  # unchanged

# -------------------- helpers --------------------

def _downsample(data, target_points=2000):
    factor = max(1, data.shape[1] // target_points)
    return data[:, ::factor], factor

def _load_existing_segmentation_flat(session_file, exp_structure):
    """
    Read the *flat* segmentation:
      "segmentation": { part: [[s,e], ...] | [] }
    Return {part: (s,e)} using the FIRST pair, if available.
    """
    try:
        with open(session_file, "r") as f:
            data = json.load(f)
    except Exception:
        return {}

    seg = data.get("segmentation", {})
    result = {}
    for part in exp_structure:
        val = seg.get(part)
        # expect [] OR [[s,e], ...]
        if isinstance(val, list) and val:
            first = val[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                try:
                    s, e = float(first[0]), float(first[1])
                except (TypeError, ValueError):
                    continue
                result[part] = (s, e)
    return result

def _format_range_se(start_end):
    """Format (s,e) -> 's - e' with one decimal; empty string if None."""
    if not start_end:
        return ""
    s, e = start_end
    return f"{s:.1f} - {e:.1f}"

_rng_re = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*[-–]\s*([+-]?\d+(?:\.\d+)?)\s*$")

def _parse_range_text(txt):
    """
    Accepts 'start-end' (spaces allowed, '-' or '–').
    Returns (start, end) as floats (ordered), or None if invalid/empty.
    """
    if not txt:
        return None
    m = _rng_re.match(txt)
    if not m:
        return None
    s, e = float(m.group(1)), float(m.group(2))
    if s > e:
        s, e = e, s
    return (s, e)

def _write_segmentation_flat(session_file, exp_structure, block_to_ranges):
    """
    Persist the flat segmentation into session_file.
    - Ensures 'segmentation' exists.
    - Ensures every block in exp_structure is present.
    - Writes list-of-pairs [[s,e]] or [].
    """
    # Load file (create minimal structure if missing)
    try:
        with open(session_file, "r") as f:
            data = json.load(f)
    except Exception:
        data = {}

    seg = data.get("segmentation", {}) or {}

    # Normalize and write
    for part in exp_structure:
        v = block_to_ranges.get(part, None)
        if v is None:
            # if not provided, keep existing, else ensure key exists as []
            if part not in seg:
                seg[part] = []
            continue

        if v == []:
            seg[part] = []
        else:
            # v expected as (s,e) or [[s,e]]
            if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                seg[part] = [[float(v[0]), float(v[1])]]
            elif isinstance(v, list) and v and isinstance(v[0], (list, tuple)) and len(v[0]) >= 2:
                s, e = float(v[0][0]), float(v[0][1])
                seg[part] = [[s, e]]
            else:
                # malformed -> leave as-is (or init empty)
                seg[part] = seg.get(part, [])

    data["segmentation"] = seg

    # Save back
    with open(session_file, "w") as f:
        json.dump(data, f, indent=2)

# -------------------- plotting --------------------

def _view_channels_bokeh_server(data, hemispheres, tms_indexes, fs=4000, range_sink=None):
    """
    Build interactive Bokeh plots for EMG channels with a RangeTool overview.
    Calls range_sink(hemi, start, end) whenever the visible x-range changes.
    """
    layouts = []
    t = np.arange(data.shape[1]) / fs
    initial_range = 320  # seconds

    for idx, hemi in enumerate(hemispheres):
        start = t[0]
        end = min(t[0] + initial_range, t[-1])
        y = data[idx] / 1000.0
        y_start, y_end = float(np.min(y)), float(np.max(y))

        # Main plot
        p = figure(
            height=300,
            title=f"{hemi} Hemisphere",
            tools="xpan,xwheel_zoom,reset,box_zoom",
            active_scroll="xwheel_zoom",
            x_range=Range1d(start=start, end=end),
            y_range=Range1d(start=y_start, end=y_end),
            y_axis_label="EMG (mV)",
            sizing_mode="stretch_width",
        )
        source_hemi = ColumnDataSource(data=dict(x=t, y=y))
        p.line('x', 'y', source=source_hemi, line_width=2)

        # Pulse markers
        for pulse_idx in tms_indexes:
            p.add_layout(Span(location=pulse_idx / fs, dimension='height',
                              line_color='gray', line_width=1,
                              line_alpha=0.5, line_dash='dashed'))

        overview = figure(
            height=120,
            tools="",                       # we'll add tools explicitly below
            toolbar_location="right",       # show toolbar so y-zoom works
            x_range=Range1d(start=t[0], end=t[-1]),
            y_range=Range1d(start=y_start, end=y_end),
            y_axis_label="EMG (mV)",
            sizing_mode="stretch_width",
        )
        overview.line('x', 'y', source=source_hemi)

        for pulse_idx in tms_indexes:
            overview.add_layout(Span(location=pulse_idx / fs, dimension='height',
                                    line_color='gray', line_width=1,
                                    line_alpha=0.5, line_dash='dashed'))

        # RangeTool still controls the main plot's x_range
        range_tool = RangeTool(x_range=p.x_range)
        range_tool.overlay.fill_color = "gray"
        range_tool.overlay.fill_alpha = 0.2

        # Y-axis zoom tools (height-only)
        y_zoom = WheelZoomTool(dimensions='height')

        # Add tools and set actives
        overview.add_tools(y_zoom, range_tool)
        overview.toolbar.active_scroll = y_zoom       # wheel zooms Y in the overview
        overview.toolbar.active_multi  = range_tool   # drag still manipulates the RangeTool

        # Reset Y-range button (resets BOTH the main plot and the overview)
        reset_button = Button(label="Reset Y-Range", width=100)
        reset_button.js_on_click(CustomJS(
            args=dict(p=p, ov=overview, y_start=y_start, y_end=y_end),
            code="""
                p.y_range.start  = y_start;
                p.y_range.end    = y_end;
                ov.y_range.start = y_start;
                ov.y_range.end   = y_end;
            """
        ))

        # Info widgets
        pulse_div = Div(text=f"<b>Visible pulses:</b> 0", width=150, style={'text-align': 'center'})
        range_div = Div(text=f"<b>Visible range:</b> {start:.1f} - {end:.1f} s", width=1200, style={'text-align': 'center'})

        # Python callback for range updates
        def update_range(attr, old, new, x_range=p.x_range, pulse_div=pulse_div,
                         range_div=range_div, hemi=hemi):
            visible_start = float(x_range.start)
            visible_end = float(x_range.end)
            # Count visible pulses
            count = sum(1 for pulse in tms_indexes if visible_start * fs <= pulse <= visible_end * fs)
            pulse_div.text = f"<b>Visible pulses:</b> {count}"
            range_div.text = f"<b>Visible range:</b> {visible_start:.2f} - {visible_end:.2f} s"

            if range_sink is not None:
                try:
                    range_sink(hemi, visible_start, visible_end)
                except Exception as e:
                    print(f"[range_sink error] {e}")

        p.x_range.on_change("start", update_range)
        p.x_range.on_change("end", update_range)

        top_row = bk_row(reset_button, pulse_div, range_div, sizing_mode="stretch_width")
        layouts.append(bk_column(top_row, p, overview, sizing_mode="stretch_width"))

        if range_sink is not None:
            try:
                range_sink(hemi, float(p.x_range.start), float(p.x_range.end))
            except Exception as e:
                print(f"[range_sink init error] {e}")

    return layouts

# -------------------- app --------------------

def start_bokeh_app(meta, session_file: str, exp_structure, SCRIPT_DIR, ranges_store, ranges_lock):
    """
    Launch a Bokeh app in a background thread; return the chosen port.
    'ranges_store' is updated by the plot sinks: {hemi: (start, end)}.
    """
    def bkapp(doc):
        # in-memory store for what we’ve already captured via "Set" or prefill
        block_ranges = {}  # {part: (s, e)}

        # sink from plots (keeps the latest visible per hemi)
        def sink(hemi, start, end):
            with ranges_lock:
                ranges_store[hemi] = (start, end)

        # data + plots
        data, tms_indexes = load_data(meta['input_file'])
        data_ds, factor = _downsample(data, target_points=10000)
        plots = _view_channels_bokeh_server(
            data_ds,
            meta["hemispheres"],
            tms_indexes // factor,
            fs=meta["sampling_rate"] / factor,
            range_sink=sink,
        )
        left_col = bk_column(*plots, sizing_mode="stretch_both")

        # ---------- right panel UI ----------
        RIGHT_LABEL_W   = 70
        RIGHT_INPUT_W   = 90
        RIGHT_BUTTON_W  = 40
        COL_GAP         = 4
        PANEL_PADDING   = 16
        SCROLLBAR_W     = 16
        HEADER_H        = 24
        ROW_H           = 28
        PANEL_H         = 400
        SAVE_BUTTON_W = RIGHT_LABEL_W + RIGHT_INPUT_W + RIGHT_BUTTON_W + 2 * COL_GAP + (2 * PANEL_PADDING)
        RIGHT_PANEL_W = RIGHT_LABEL_W + RIGHT_INPUT_W + RIGHT_BUTTON_W + 2 * COL_GAP + (2 * PANEL_PADDING) + SCROLLBAR_W

        doc.add_root(Div(text=f"""
        <style>
          .scrollpane {{
            overflow-y:auto; overflow-x:hidden;
            box-sizing:border-box;
            padding:{PANEL_PADDING}px {PANEL_PADDING+SCROLLBAR_W}px {PANEL_PADDING}px {PANEL_PADDING}px;
            scrollbar-gutter:stable;
          }}
          .lab {{ text-align:right; margin:0; }}
          .tight {{ margin:0; padding:0; }}
        </style>
        """, width=0, height=0))

        title_div = Div(text="<h3 class='tight'>Experiment Blocks</h3>", align='center')

        hdr = bk_row(
            Div(text="<b>Block</b>", css_classes=["lab"], width=RIGHT_LABEL_W, height=HEADER_H),
            Div(text="<b>Range (s)</b>", width=RIGHT_INPUT_W, height=HEADER_H, align='center'),
            Div(text="&nbsp;", width=RIGHT_BUTTON_W, height=HEADER_H),
            sizing_mode="fixed", spacing=COL_GAP,
        )

        rows = [hdr]
        text_inputs = {}  # part -> TextInput

        # prefill from flat segmentation
        prefill_map = _load_existing_segmentation_flat(session_file, exp_structure)
        block_ranges.update(prefill_map)

        def make_set_callback(part: str, ti: TextInput):
            def _cb():
                # Snapshot current visible windows from all hemispheres:
                # use shared (min start, max end) as a single range for the block.
                with ranges_lock:
                    current = dict(ranges_store)  # {hemi: (s,e)}
                if current:
                    starts = [se[0] for se in current.values()]
                    ends   = [se[1] for se in current.values()]
                    shared = (min(starts), max(ends))
                    block_ranges[part] = shared
                    ti.value = _format_range_se(shared)
            return _cb

        for part in exp_structure:
            lab = Div(text=f"<b>{part}</b>", css_classes=["lab"], width=RIGHT_LABEL_W, height=ROW_H)
            prefilled_value = _format_range_se(prefill_map.get(part))
            ti  = TextInput(placeholder="e.g. 100 - 240",
                            value=prefilled_value,
                            width=RIGHT_INPUT_W, height=ROW_H)
            btn = Button(label="Set", button_type="default", width=RIGHT_BUTTON_W, height=ROW_H)
            btn.on_click(make_set_callback(part, ti))
            rows.append(bk_row(lab, ti, btn, sizing_mode="fixed", spacing=COL_GAP))
            text_inputs[part] = ti

        scroll_body = bk_column(*rows, sizing_mode="stretch_width", spacing=0)
        scroll_container = bk_column(scroll_body, width=RIGHT_PANEL_W, height=PANEL_H, sizing_mode="fixed")
        scroll_container.css_classes = ["scrollpane"]

        DEFAULT_SAVE_LABEL = "Save segmentation"
        save_btn = Button(label=DEFAULT_SAVE_LABEL,
                          button_type="warning",
                          width=SAVE_BUTTON_W, height=28,
                          align="start")

        _status_counter = {"v": 0}

        def flash_status(text: str, typ: str = "success", duration_ms: int = 1500):
            _status_counter["v"] += 1
            my_token = _status_counter["v"]
            save_btn.label = text
            save_btn.button_type = typ
            def _reset():
                if _status_counter["v"] == my_token:
                    save_btn.label = DEFAULT_SAVE_LABEL
                    save_btn.button_type = "warning"  # reset to original style
            doc.add_timeout_callback(_reset, duration_ms)

        def save_all():
            """
            Flat schema write:
            - empty field -> []
            - valid "start - end" -> [[s,e]] (handled by writer from [s,e])
            - invalid non-empty -> clear ([]) and warn
            """
            to_save = {}
            invalid_parts = []

            for part in exp_structure:
                txt = text_inputs[part].value.strip() if part in text_inputs else ""
                if txt == "":
                    to_save[part] = []  # explicit clear
                else:
                    parsed = _parse_range_text(txt)  # -> (s,e) or None
                    if parsed:
                        to_save[part] = list(parsed)  # (s,e) -> [s,e]; writer normalizes to [[s,e]]
                    else:
                        invalid_parts.append(part)
                        to_save[part] = []  # choose to clear on invalid input

            try:
                write_segmentation_ranges(
                    session_file=session_file,
                    block_ranges=to_save,
                    hemis=meta.get("hemispheres", []),  # ignored for flat schema (kept for API compat)
                    exp_structure=list(exp_structure),
                )

                # refresh in-memory view so the UI reflects what’s on disk
                new_prefill = _load_existing_segmentation_flat(session_file, exp_structure)
                block_ranges.clear()
                block_ranges.update(new_prefill)

                if invalid_parts:
                    flash_status(f"Saved (check: {', '.join(invalid_parts)})", "warning", 2200)
                else:
                    flash_status("Saved ✅", "success", 1200)

            except Exception as e:
                # import traceback; print(traceback.format_exc())
                flash_status(f"Error: {e}", "danger", 2200)

        save_btn.on_click(save_all)

        right_col = bk_column(title_div, scroll_container, save_btn,
                              width=RIGHT_PANEL_W, height=PANEL_H,
                              sizing_mode="fixed", spacing=6)

        doc.add_root(bk_row(left_col, right_col, sizing_mode="stretch_both", spacing=10))

    # ---------- start server ----------
    port_holder = []

    def run_server(holder):
        server = Server({'/bkapp': bkapp}, port=0, allow_websocket_origin=["*"], address="127.0.0.1", use_xheaders=True)
        server.start()
        holder.append(server.port)
        server.io_loop.start()

    t = threading.Thread(target=run_server, args=(port_holder,), daemon=True)
    t.start()
    while not port_holder:
        time.sleep(0.1)
    return port_holder[0]
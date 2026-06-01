# app/bk_embedding/segmentation.py
# -*- coding: utf-8 -*-
"""
app/bk_embedding/segmentation.py
---------------------------------
Creates and runs the embedded Bokeh server.

Accepts:
  active_blocks     — only these blocks appear in the right panel.
  downsample_factor — 1 = full resolution; N = keep every Nth sample.
"""

from __future__ import annotations

import time
import threading
import json
import re
from typing import Any

import numpy as np

from bokeh.server.server import Server
from bokeh.plotting import figure
from bokeh.layouts import column as bk_column, row as bk_row
from bokeh.models import (
    ColumnDataSource,
    RangeTool,
    Button,
    Div,
    Range1d,
    TextInput,
    CustomJS,
    WheelZoomTool,
)

from app.utils.persistence import write_segmentation_ranges
from app.utils.tms_module import _load_data_cached


# -------------------- helpers --------------------

def _apply_downsample(data: np.ndarray, factor: int) -> tuple[np.ndarray, int]:
    """Return (data[:,::factor], factor). Factor 1 returns original unchanged."""
    if factor <= 1:
        return data, 1
    return data[:, ::factor], int(factor)


def _template_requests_emg_ref(meta: dict) -> bool:
    tpath = meta.get("template_file")
    if not tpath:
        return False
    try:
        with open(tpath, "r") as f:
            tpl = json.load(f)
    except Exception:
        return False
    other = tpl.get("other", {}) or {}
    return str(other.get("include_rest_emg_ref", "")).strip().lower() == "yes"


def _maybe_add_emg_ref(meta: dict, exp_structure: list[str]) -> list[str]:
    exp_structure = list(exp_structure or [])
    if _template_requests_emg_ref(meta) and "emg_ref" not in exp_structure:
        exp_structure.append("emg_ref")
    return exp_structure


def _load_existing_segmentation_flat(
    session_file: str, exp_structure: list[str]
) -> dict[str, tuple[float, float]]:
    try:
        with open(session_file, "r") as f:
            data = json.load(f)
    except Exception:
        return {}
    seg    = data.get("segmentation", {}) or {}
    result: dict[str, tuple[float, float]] = {}
    for part in exp_structure:
        val = seg.get(part)
        if isinstance(val, list) and val:
            first = val[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                try:
                    s, e = float(first[0]), float(first[1])
                except (TypeError, ValueError):
                    continue
                result[part] = (s, e)
    return result


def _format_range_se(start_end: tuple[float, float] | None) -> str:
    if not start_end:
        return ""
    return f"{start_end[0]:.1f} - {start_end[1]:.1f}"


_rng_re = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*[-–]\s*([+-]?\d+(?:\.\d+)?)\s*$")


def _parse_range_text(txt: str) -> tuple[float, float] | None:
    if not txt:
        return None
    m = _rng_re.match(txt)
    if not m:
        return None
    s, e = float(m.group(1)), float(m.group(2))
    if s > e:
        s, e = e, s
    return (s, e)


# -------------------- plotting --------------------

def _view_channels_bokeh_server(
    data: np.ndarray,
    hemispheres: list[str],
    tms_indexes: np.ndarray,
    fs: float = 4000.0,
    range_sink=None,
):
    """
    Build interactive EMG overview plots.
    Uses a single Segment glyph for TMS pulse markers (fast at high pulse counts).
    """
    layouts      = []
    t            = np.arange(data.shape[1]) / fs
    initial_range = 320  # seconds

    for idx, hemi in enumerate(hemispheres):
        start = float(t[0])
        end   = float(min(t[0] + initial_range, t[-1]))

        y       = data[idx] / 1000.0
        y_start = float(np.min(y))
        y_end   = float(np.max(y))

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
        p.line("x", "y", source=source_hemi, line_width=2)

        if len(tms_indexes) > 0:
            pulse_t = (tms_indexes / fs).tolist()
            n_p     = len(pulse_t)
            p.segment(
                x0=pulse_t, y0=[y_start] * n_p,
                x1=pulse_t, y1=[y_end]   * n_p,
                line_color="gray", line_width=1, line_alpha=0.5, line_dash="dashed",
            )

        overview = figure(
            height=120,
            tools="",
            toolbar_location="right",
            x_range=Range1d(start=float(t[0]), end=float(t[-1])),
            y_range=Range1d(start=y_start, end=y_end),
            y_axis_label="EMG (mV)",
            sizing_mode="stretch_width",
        )
        overview.line("x", "y", source=source_hemi)

        if len(tms_indexes) > 0:
            pulse_t = (tms_indexes / fs).tolist()
            n_p     = len(pulse_t)
            overview.segment(
                x0=pulse_t, y0=[y_start] * n_p,
                x1=pulse_t, y1=[y_end]   * n_p,
                line_color="gray", line_width=1, line_alpha=0.5, line_dash="dashed",
            )

        range_tool = RangeTool(x_range=p.x_range)
        range_tool.overlay.fill_color = "gray"
        range_tool.overlay.fill_alpha = 0.2

        y_zoom = WheelZoomTool(dimensions="height")
        overview.add_tools(y_zoom, range_tool)
        overview.toolbar.active_scroll = y_zoom
        overview.toolbar.active_multi  = range_tool

        reset_button = Button(label="Reset Y-Range", width=100)
        reset_button.js_on_click(
            CustomJS(
                args=dict(p=p, ov=overview, y_start=y_start, y_end=y_end),
                code="""
                    p.y_range.start  = y_start;
                    p.y_range.end    = y_end;
                    ov.y_range.start = y_start;
                    ov.y_range.end   = y_end;
                """,
            )
        )

        pulse_div = Div(text="<b>Visible pulses:</b> 0", width=150,
                        style={"text-align": "center"})
        range_div = Div(
            text=f"<b>Visible range:</b> {start:.1f} - {end:.1f} s",
            width=1200, style={"text-align": "center"},
        )

        def update_range(attr, old, new, x_range=p.x_range, pulse_div=pulse_div,
                         range_div=range_div, hemi=hemi):
            visible_start = float(x_range.start)
            visible_end   = float(x_range.end)
            count = sum(1 for pulse in tms_indexes
                        if visible_start * fs <= pulse <= visible_end * fs)
            pulse_div.text = f"<b>Visible pulses:</b> {count}"
            range_div.text = f"<b>Visible range:</b> {visible_start:.2f} - {visible_end:.2f} s"
            if range_sink is not None:
                try:
                    range_sink(hemi, visible_start, visible_end)
                except Exception as e:
                    print(f"[range_sink error] {e}")

        p.x_range.on_change("start", update_range)
        p.x_range.on_change("end",   update_range)

        top_row = bk_row(reset_button, pulse_div, range_div, sizing_mode="stretch_width")
        layouts.append(bk_column(top_row, p, overview, sizing_mode="stretch_width"))

        if range_sink is not None:
            try:
                range_sink(hemi, float(p.x_range.start), float(p.x_range.end))
            except Exception as e:
                print(f"[range_sink init error] {e}")

    return layouts


# -------------------- app --------------------

def start_bokeh_app(
    meta,
    session_file: str,
    exp_structure,
    SCRIPT_DIR,
    ranges_store,
    ranges_lock,
    active_blocks: list[str] | None = None,
    downsample_factor: int = 5,
):
    """
    Launch a Bokeh app in a background thread; return the chosen port.

    active_blocks:     blocks shown in the right panel (and validated on advance).
                       Defaults to all blocks in exp_structure.
    downsample_factor: 1 = full resolution; N = keep every Nth sample.
    """

    def bkapp(doc):
        exp_structure_local = _maybe_add_emg_ref(meta, exp_structure)

        # Right panel only shows active blocks; emg_ref is always included.
        panel_blocks = list(active_blocks) if active_blocks is not None else list(exp_structure_local)
        # Ensure emg_ref is in the panel list if it's in the full structure.
        if "emg_ref" in exp_structure_local and "emg_ref" not in panel_blocks:
            panel_blocks.append("emg_ref")

        block_ranges: dict[str, tuple[float, float]] = {}

        def sink(hemi: str, start: float, end: float):
            with ranges_lock:
                ranges_store[hemi] = (start, end)

        channels       = meta.get("channels") or [3, 1, 2]
        channels_tuple = tuple(int(c) for c in channels)
        fs_int         = int(meta["sampling_rate"])

        data, tms_indexes = _load_data_cached(str(meta["input_file"]), channels_tuple, fs_int)

        # Apply downsample_factor (1 = unchanged).
        data_ds, applied_factor = _apply_downsample(data, int(downsample_factor))
        tms_ds = (tms_indexes // applied_factor) if applied_factor > 1 else tms_indexes

        plots = _view_channels_bokeh_server(
            data_ds,
            meta["hemispheres"],
            tms_ds,
            fs=float(meta["sampling_rate"]) / applied_factor,
            range_sink=sink,
        )
        left_col = bk_column(*plots, sizing_mode="stretch_both")

        # ---------- right panel ----------
        RIGHT_LABEL_W  = 70
        RIGHT_INPUT_W  = 90
        RIGHT_BUTTON_W = 40
        COL_GAP        = 4
        PANEL_PADDING  = 16
        SCROLLBAR_W    = 16
        HEADER_H       = 24
        ROW_H          = 28
        PANEL_H        = 400

        SAVE_BUTTON_W  = RIGHT_LABEL_W + RIGHT_INPUT_W + RIGHT_BUTTON_W + 2 * COL_GAP + (2 * PANEL_PADDING)
        RIGHT_PANEL_W  = SAVE_BUTTON_W + SCROLLBAR_W

        doc.add_root(
            Div(
                text=f"""
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
                """,
                width=0, height=0,
            )
        )

        title_div = Div(text="<h3 class='tight'>Experiment Blocks</h3>", align="center")
        hdr = bk_row(
            Div(text="<b>Block</b>",      css_classes=["lab"], width=RIGHT_LABEL_W, height=HEADER_H),
            Div(text="<b>Range (s)</b>",  width=RIGHT_INPUT_W, height=HEADER_H, align="center"),
            Div(text="&nbsp;",            width=RIGHT_BUTTON_W, height=HEADER_H),
            sizing_mode="fixed", spacing=COL_GAP,
        )

        rows        : list = [hdr]
        text_inputs : dict[str, TextInput] = {}

        prefill_map = _load_existing_segmentation_flat(session_file, panel_blocks)
        block_ranges.update(prefill_map)

        def make_set_callback(part: str, ti: TextInput):
            def _cb():
                with ranges_lock:
                    current = dict(ranges_store)
                if current:
                    starts = [se[0] for se in current.values()]
                    ends   = [se[1] for se in current.values()]
                    shared = (min(starts), max(ends))
                    block_ranges[part] = shared
                    ti.value = _format_range_se(shared)
            return _cb

        # Only show panel_blocks in the right panel UI.
        for part in panel_blocks:
            lab = Div(text=f"<b>{part}</b>", css_classes=["lab"],
                      width=RIGHT_LABEL_W, height=ROW_H)
            prefilled = _format_range_se(prefill_map.get(part))
            ti  = TextInput(placeholder="e.g. 100 - 240", value=prefilled,
                            width=RIGHT_INPUT_W, height=ROW_H)
            btn = Button(label="Set", button_type="default",
                         width=RIGHT_BUTTON_W, height=ROW_H)
            btn.on_click(make_set_callback(part, ti))
            rows.append(bk_row(lab, ti, btn, sizing_mode="fixed", spacing=COL_GAP))
            text_inputs[part] = ti

        scroll_body      = bk_column(*rows, sizing_mode="stretch_width", spacing=0)
        scroll_container = bk_column(scroll_body, width=RIGHT_PANEL_W, height=PANEL_H,
                                     sizing_mode="fixed")
        scroll_container.css_classes = ["scrollpane"]

        DEFAULT_SAVE_LABEL = "Save segmentation"
        save_btn = Button(label=DEFAULT_SAVE_LABEL, button_type="warning",
                          width=SAVE_BUTTON_W, height=28, align="start")

        _status_counter = {"v": 0}

        def flash_status(text: str, typ: str = "success", duration_ms: int = 1500):
            _status_counter["v"] += 1
            my_token = _status_counter["v"]
            save_btn.label       = text
            save_btn.button_type = typ

            def _reset():
                if _status_counter["v"] == my_token:
                    save_btn.label       = DEFAULT_SAVE_LABEL
                    save_btn.button_type = "warning"

            doc.add_timeout_callback(_reset, duration_ms)

        def save_all():
            to_save      : dict[str, Any] = {}
            invalid_parts: list[str]      = []

            for part in panel_blocks:
                txt = text_inputs[part].value.strip() if part in text_inputs else ""
                if txt == "":
                    to_save[part] = []
                else:
                    parsed = _parse_range_text(txt)
                    if parsed:
                        to_save[part] = [float(parsed[0]), float(parsed[1])]
                    else:
                        invalid_parts.append(part)
                        to_save[part] = []

            try:
                write_segmentation_ranges(
                    session_file=session_file,
                    block_ranges=to_save,
                    hemis=meta.get("hemispheres", []),
                    exp_structure=list(exp_structure_local),
                )
                new_prefill = _load_existing_segmentation_flat(session_file, panel_blocks)
                block_ranges.clear()
                block_ranges.update(new_prefill)

                if invalid_parts:
                    flash_status(f"Saved (check: {', '.join(invalid_parts)})", "warning", 2200)
                else:
                    flash_status("Saved ✅", "success", 1200)
            except Exception as e:
                flash_status(f"Error: {e}", "danger", 2200)

        save_btn.on_click(save_all)

        right_col = bk_column(
            title_div, scroll_container, save_btn,
            width=RIGHT_PANEL_W, height=PANEL_H,
            sizing_mode="fixed", spacing=6,
        )

        doc.add_root(bk_row(left_col, right_col, sizing_mode="stretch_both", spacing=10))

    # ---------- start server ----------
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
        raise RuntimeError("Segmentation Bokeh server failed to start within 5 seconds.")

    return port_holder[0]

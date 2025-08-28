"""
bokeh_embed.py
--------------
Creates and runs the embedded Bokeh server and right-hand controls.

- Shows EMG channels with a RangeTool
- Lets user pick a visible window per hemisphere and "Set" it for each block
- Saves those selections into the subject/session JSON under 'segmentation'
"""
import time, threading, os, json, numpy as np
from bokeh.server.server import Server
from bokeh.layouts import column as bk_column, row as bk_row
from bokeh.models import Div, TextInput, Button, Range1d
import tmp_module as tms_utils
from utils.persistence import write_segmentation_ranges

def _downsample(data, target_points=2000):
    factor = max(1, data.shape[1] // target_points)
    return data[:, ::factor], factor

def start_bokeh_app(meta, session_file: str, exp_structure, SCRIPT_DIR, ranges_store, ranges_lock):
    """
    Launch a Bokeh app in a background thread; return the chosen port.
    'ranges_store' is a shared dict updated by range sinks: {hemi: (start, end)}.
    """
    def bkapp(doc):
        # ---------- live stores ----------
        block_ranges = {}  # {part: {hemi: (start, end)}}

        # ---------- sink from plots ----------
        def sink(hemi, start, end):
            with ranges_lock:
                ranges_store[hemi] = (start, end)

        # ---------- data + plots ----------
        data, tms_indexes = tms_utils.load_data(meta['input_file'])
        data_ds, factor = _downsample(data, target_points=10000)
        plots = tms_utils.view_channels_bokeh_server(
            data_ds,
            meta["hemispheres"],
            tms_indexes // factor,
            fs=meta["sampling_rate"] / factor,
            range_sink=sink,
        )
        left_col = bk_column(*plots, sizing_mode="stretch_both")

        # ---------- right panel UI ----------
        RIGHT_LABEL_W   = 90
        RIGHT_INPUT_W   = 160
        RIGHT_BUTTON_W  = 70
        COL_GAP         = 4
        PANEL_PADDING   = 16
        SCROLLBAR_W     = 16
        HEADER_H        = 24
        ROW_H           = 28
        PANEL_H         = 400
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

        title_div = Div(text="<h3 class='tight'>Experiment Blocks</h3>")

        hdr = bk_row(
            Div(text="<b>Block</b>", css_classes=["lab"], width=RIGHT_LABEL_W, height=HEADER_H),
            Div(text="<b>Ranges (s)</b>", width=RIGHT_INPUT_W, height=HEADER_H),
            Div(text="&nbsp;", width=RIGHT_BUTTON_W, height=HEADER_H),
            sizing_mode="fixed", spacing=COL_GAP,
        )

        rows = [hdr]
        text_inputs = {}  # part -> TextInput

        def make_set_callback(part: str, ti: TextInput):
            def _cb():
                # snapshot current visible range for each hemi
                with ranges_lock:
                    current = dict(ranges_store)
                block_ranges[part] = current  # {hemi: (start, end)}
                # format for display
                txt = ", ".join(f"{h}:{s:.1f}-{e:.1f}" for h, (s, e) in current.items())
                ti.value = txt
            return _cb

        for part in exp_structure:
            lab = Div(text=f"<b>{part}</b>", css_classes=["lab"], width=RIGHT_LABEL_W, height=ROW_H)
            ti  = TextInput(placeholder="e.g. left:0-240, right:0-240", width=RIGHT_INPUT_W, height=ROW_H)
            btn = Button(label="Set", button_type="primary", width=RIGHT_BUTTON_W, height=ROW_H)
            btn.on_click(make_set_callback(part, ti))
            rows.append(bk_row(lab, ti, btn, sizing_mode="fixed", spacing=COL_GAP))
            text_inputs[part] = ti

        scroll_body = bk_column(*rows, sizing_mode="stretch_width", spacing=0)
        scroll_container = bk_column(scroll_body, width=RIGHT_PANEL_W, height=PANEL_H, sizing_mode="fixed")
        scroll_container.css_classes = ["scrollpane"]

        save_btn = Button(label="Save selections to session JSON", button_type="success", width=RIGHT_PANEL_W, height=28)

        def save_all():
            # Write to sub-<id>_ses-<n>.json under 'segmentation'
            try:
                write_segmentation_ranges(
                    session_file=session_file,
                    block_ranges=block_ranges,
                    hemis=list(meta.get("hemispheres", [])),
                    exp_structure=list(exp_structure),
                )
                save_btn.label = "Saved ✅"
            except Exception as e:
                save_btn.label = f"Error: {e}"

        save_btn.on_click(save_all)

        right_col = bk_column(title_div, scroll_container, save_btn, width=RIGHT_PANEL_W, sizing_mode="fixed", spacing=6)

        doc.add_root(bk_row(left_col, right_col, sizing_mode="stretch_both", spacing=10))

    # ---------- start server ----------
    port_holder = []
    def run_server(holder):
        server = Server({'/bkapp': bkapp}, port=0, allow_websocket_origin=["*"], address="127.0.0.1", use_xheaders=True)
        server.start(); holder.append(server.port); server.io_loop.start()

    t = threading.Thread(target=run_server, args=(port_holder,), daemon=True)
    t.start()
    while not port_holder: time.sleep(0.1)
    return port_holder[0]

# -*- coding: utf-8 -*-
"""
to activate conda env:
        conda activate cfom_mep_preprocessing

to execute on command line:
        
        streamlit run main_gui.py
"""

import os
import time
import json
import threading
import numpy as np

import streamlit as st
import streamlit.components.v1 as components

from bokeh.server.server import Server
from bokeh.layouts import column as bk_column

import tmp_module as tms_utils

#%% Configure Layout  

st.set_page_config(layout="wide")
# st.markdown("""
# <style>
# /* Reduce side padding */
# .block-container { padding-left: 1rem; padding-right: 1rem; }
# </style>
# """, unsafe_allow_html=True)

#%% Style functions

def render_text(
    text,
    font_color="black",
    font_weight="normal",          # normal or bold
    horizontal_alignment="right",  # left, center, right
    font_size=None,                # CSS font size (if None, use heading map)
    nowrap=True,                   # True = do not wrap, False = allow wrapping
    heading_level=None             # None for normal text, or 1–5 for heading style
):
    """
    Render styled text or headings in Streamlit using HTML.

    Args:
        text (str): Text to display.
        font_color (str): Text color.
        font_weight (str): 'normal' or 'bold'.
        horizontal_alignment (str): 'left', 'center', 'right'.
        font_size (str|None): CSS font size (e.g., '1.25rem', '20px').
                              If None and heading_level is set, a default size is used.
        nowrap (bool): If True, text will not wrap.
        heading_level (int|None): Optional heading level (1–5).
                                  If None, behaves like styled text.
    """
    justify_map = {"left": "flex-start", "center": "center", "right": "flex-end"}
    h_align = justify_map.get(horizontal_alignment.lower(), "flex-end")
    white_space = "nowrap" if nowrap else "normal"

    # Default font sizes for heading levels (you can tweak as needed)
    heading_sizes = {
        1: "2rem",     # ~32px
        2: "1.75rem",  # ~28px
        3: "1.5rem",   # ~24px
        4: "1.25rem",  # ~20px
        5: "1rem",     # ~16px
    }

    # Pick font size
    if heading_level in heading_sizes and font_size is None:
        font_size = heading_sizes[heading_level]
        font_weight = "bold"  # headings are bold by default

    html_code = (
        f"<div style='display: flex; width: 100%; justify-content: {h_align}; "
        f"font-weight: {font_weight}; color: {font_color}; font-size: {font_size}; "
        f"white-space: {white_space}; margin: 0.25em 0;'>"
        f"{text}</div>"
    )

    st.markdown(html_code, unsafe_allow_html=True)


#%% Usefull functions

def downsample(data, target_points=2000):
    factor = max(1, data.shape[1] // target_points)
    return data[:, ::factor], factor

#%% main code

# -------------------------
# Get script directory and default template path
# -------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEMPLATE = os.path.join(SCRIPT_DIR, "experiment_template.json")
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "example_data.bin")

# -------------------------
# Initialize session state
# -------------------------
if "step" not in st.session_state:
    # st.session_state.step = "input"
    st.session_state.step = "proceed"
if "metadata" not in st.session_state:
    st.session_state.metadata = {
        "template_file": DEFAULT_TEMPLATE,
        "input_file": DEFAULT_INPUT,
        "sampling_rate": 4000,
        "subj_id": "example_sub",
        "session": 1,
        "hemispheres": ["left"]
    }

meta = st.session_state.metadata

# --- Step 1: Input form ---
if st.session_state.step == "input":
    st.title("TMS-EMG Preprocessing GUI")

    # Experiment template
    meta["template_file"] = st.text_input(
        "Select experiment template file", value=meta["template_file"]
    )

    # EMG file: type path or browse
    meta["input_file"] = st.text_input(
        "Select EMG file", value=meta["input_file"]
    )

    
    # Arrange the rest in columns
    col1, col2, col3, col4, col5,  = st.columns([2, 1, 1, 1, 3])  # tweak ratios as needed

    with col1:
        meta["subj_id"] = st.text_input("ID", value=meta["subj_id"])
    with col2:
        meta["session"] = st.number_input(
            "Session", min_value=1, value=meta["session"], step=1
        )
    with col3:
        meta["sampling_rate"] = st.number_input(
            "Sampling Rate (Hz)", value=meta["sampling_rate"], step=100
        )
    with col4:
        st.write("Hemisphere(s):")
        hemi_col1, hemi_col2 = st.columns(2)  # two mini-columns
        with hemi_col1:
            left = st.checkbox("Left", value="left" in meta["hemispheres"] or not meta["hemispheres"])
        with hemi_col2:
            right = st.checkbox("Right", value="right" in meta["hemispheres"])

    # Update hemisphere list
    meta["hemispheres"] = []
    if left:
        meta["hemispheres"].append("left")
    if right:
        meta["hemispheres"].append("right")

    if st.button("Advance"):
        # validation checks
        if not os.path.exists(meta["template_file"]):
            st.error("Template file does not exist!")
        elif not os.path.exists(meta["input_file"]):
            st.error("EMG file does not exist!")
        elif not meta["hemispheres"]:
            st.error("Select at least one hemisphere!")
        else:
            st.session_state.step = "confirm"
            st.rerun()

# --- Step 2: Confirmation "popup" ---
elif st.session_state.step == "confirm":
    st.subheader("Please confirm your choices:")

    meta = st.session_state.metadata

    # Update experiment info from template
    with open(meta["template_file"], "r") as f:
        template = json.load(f)

    meta['exp_name'] = template["experiment_name"]
    meta['channels'] = np.array([
        template["channels"]["synch_pulse"],
        template["channels"]["right"],
        template["channels"]["left"]
    ])
    meta['exp_structure'] = template["experiment_structure"]
    # --------------------------------------

    hemi_str1 = " and ".join(meta["hemispheres"]) if meta["hemispheres"] else "none"
    hemi_str2 = 's' if 'and' in hemi_str1 else ''
    st.info(
        f"Process file `{meta['input_file']}` of `{meta['exp_name']}` experiment. .\n\n"
        f"This data will be associated to subject `{meta['subj_id']}`, session `{meta['session']}` "
        f"with stimulation in the **{hemi_str1}** hemisphere(s).\n\nProceed?"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("No, go back"):
            st.session_state.step = "input"
            st.rerun()
    with col2:
        if st.button("Yes, proceed"):
            st.session_state.step = "proceed"
            st.rerun()

# --- Step 3: Proceed ---

elif st.session_state.step == "proceed":
    from bokeh.layouts import row as bk_row, column as bk_column
    from bokeh.models import Div, TextInput, Button, Spacer
    # ---- Shared store for live ranges (thread-safe) ----
    ranges_lock = threading.Lock()
    ranges_store = {}  # e.g., {"Left": (start, end), "Right": (start, end)}

    def setRange(part=None):
        """Store current ranges and also populate the matching text_input"""
        if 'block_ranges' not in st.session_state:
            st.session_state['block_ranges'] = {}
        with ranges_lock:
            current_ranges = dict(ranges_store)
        st.session_state['block_ranges'][part or 'unnamed'] = current_ranges

        # Also update the corresponding text_input value
        if part is not None:
            # format ranges as a simple string, e.g. Left: 0-240, Right: 0-240
            ranges_str = ", ".join(
                f"{s:.1f}-{e:.1f}" for hemi, (s, e) in current_ranges.items()
            )
            st.session_state[f"input_{part}"] = ranges_str

    # st.title("EMG Segmentation")
    render_text(
            "EMG Segmentation",
            font_color="black",
            font_weight="normal",          # normal or bold
            horizontal_alignment="center",  # left, center, right
            font_size=None,                # CSS font size (if None, use heading map)
            nowrap=True,                   # True = do not wrap, False = allow wrapping
            heading_level=1             # None for normal text, or 1–5 for heading style
        )

    # Load template
    with open(meta["template_file"], "r") as f:
        template = json.load(f)

    meta['exp_name'] = template["experiment_name"]
    meta['channels'] = np.array([
        template["channels"]["synch_pulse"],
        template["channels"]["right"],
        template["channels"]["left"]
    ])
    meta['exp_structure'] = template["experiment_structure"]

    
    num_channels = len(meta["hemispheres"])
    per_channel_px = 800
    iframe_height = per_channel_px * max(1, num_channels)
    iframe_width = 3000

    # ---- Bokeh server app ----
    def bkapp(doc):
        # ---------- SIZING CONTROLS ----------
        RIGHT_LABEL_W   = 80     # px
        RIGHT_INPUT_W   = 140    # px
        RIGHT_BUTTON_W  = 60     # px
        COL_GAP         = 3      # px gap between columns (2 gaps per row)
        PANEL_PADDING   = 18     # px padding for the scroll pane (top/bottom/left)
        SCROLLBAR_W     = 16     # px reserved for vertical scrollbar (Win ~16-18; macOS overlay -> set 0)
        HEADER_H        = 24
        ROW_H           = 28
        BODY_SPACING    = 0      # vertical spacing between stacked rows
        # TITLE_BOTTOM_SP = 0
        PANEL_H         = 400    # visible height of right panel (scrolls vertically)
        SAVE_BUTTON_HEIGHT = 28

        # Content width is the sum of columns + the two gaps between them
        CONTENT_W = RIGHT_LABEL_W + RIGHT_INPUT_W + RIGHT_BUTTON_W + 2 * COL_GAP
        # Final scroll container width includes padding (L+R) and the scrollbar gutter
        RIGHT_PANEL_W = CONTENT_W + (2 * PANEL_PADDING) + SCROLLBAR_W

        # ---------- live ranges ----------
        block_ranges = {}

        def sink(hemi, start, end):
            with ranges_lock:
                ranges_store[hemi] = (start, end)

        # ---------- data + plots (left) ----------
        data, tms_indexes = tms_utils.load_data(meta['input_file'])
        data_ds, factor = downsample(data, target_points=10000)

        plots = tms_utils.view_channels_bokeh_server(
            data_ds,
            meta["hemispheres"],
            tms_indexes // factor,
            fs=meta["sampling_rate"] / factor,
            range_sink=sink,
        )
        left_col = bk_column(*plots, sizing_mode="stretch_both")

        # ---------- CSS (reserve scrollbar gutter, avoid horizontal scroll) ----------
        doc.add_root(Div(text=f"""
        <style>
        .scrollpane {{
            overflow-y: auto;
            overflow-x: hidden;
            box-sizing: border-box;
            /* top right bottom left padding: add scrollbar width on the right */
            padding: {PANEL_PADDING}px {PANEL_PADDING + SCROLLBAR_W}px {PANEL_PADDING}px {PANEL_PADDING}px;
            border: 0;
            border-radius: 0;
            /* keeps a stable gutter even when content height changes */
            scrollbar-gutter: stable;
        }}
        .lab   {{ text-align: right; padding-right: 0; margin: 0; }}
        .tight {{ margin: 0; padding: 0; }}
        </style>
        """, width=0, height=0))

        # ---------- right panel ----------
        title_div = Div(text="<h3 class='tight'>Experiment Blocks</h3>")

        hdr = bk_row(
            Div(text="<b>Block</b>",      css_classes=["lab"], width=RIGHT_LABEL_W,  height=HEADER_H),
            Div(text="<b>Ranges (s)</b>",                    width=RIGHT_INPUT_W,  height=HEADER_H),
            Div(text="&nbsp;",                               width=RIGHT_BUTTON_W, height=HEADER_H),
            sizing_mode="fixed",
            spacing=COL_GAP,
        )

        rows = [hdr]

        def make_set_callback(part, text_input):
            def _cb():
                with ranges_lock:
                    current = dict(ranges_store)
                block_ranges[part] = current
                text_input.value = ", ".join(f"{s:.1f}-{e:.1f}" for _, (s, e) in current.items())
            return _cb

        for part in meta['exp_structure']:
            lab = Div(text=f"<b>{part}</b>", css_classes=["lab"], width=RIGHT_LABEL_W,  height=ROW_H)
            ti  = TextInput(placeholder="e.g. 0.0-240.0, 5.0-30.0", width=RIGHT_INPUT_W,  height=ROW_H)
            btn = Button(label="Set", button_type="primary",         width=RIGHT_BUTTON_W, height=ROW_H)
            btn.on_click(make_set_callback(part, ti))
            rows.append(bk_row(lab, ti, btn, sizing_mode="fixed", spacing=COL_GAP))

        scroll_body = bk_column(*rows, sizing_mode="stretch_width", spacing=BODY_SPACING)
        scroll_container = bk_column(
            scroll_body,
            width=RIGHT_PANEL_W,    # ← includes scrollbar gutter
            height=PANEL_H,
            sizing_mode="fixed",
        )
        scroll_container.css_classes = ["scrollpane"]

        save_btn = Button(label="Save selections to JSON", button_type="success", width=RIGHT_PANEL_W, height = SAVE_BUTTON_HEIGHT)

        def save_all():
            out = {p: {h: [float(s), float(e)] for h, (s, e) in rr.items()} for p, rr in block_ranges.items()}
            out_path = os.path.join(SCRIPT_DIR, "block_ranges.json")
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            save_btn.label = "Saved ✅"

        save_btn.on_click(save_all)

        right_col = bk_column(
            title_div,
            # Spacer(height=TITLE_BOTTOM_SP),
            scroll_container,
            save_btn,
            width=RIGHT_PANEL_W,
            sizing_mode="fixed",
            spacing=6,
        )

        doc.add_root(bk_row(left_col, right_col, sizing_mode="stretch_both", spacing=10))

    # Detect Streamlit port for websocket allowlist
    streamlit_port = os.environ.get("STREAMLIT_SERVER_PORT", "8501")

    # Start Bokeh server in background
    def run_bokeh_server(port_container):
        server = Server(
            {'/bkapp': bkapp},
            port=0,                         # let Bokeh pick a free port
            allow_websocket_origin=["*"],   # allow any origin (LOCAL DEV ONLY)
            address="127.0.0.1",            # keep it bound to localhost
            use_xheaders=True,
        )
        server.start()
        port_container.append(server.port)
        server.io_loop.start()

    port_container = []
    t = threading.Thread(target=run_bokeh_server, args=(port_container,), daemon=True)
    t.start()
    while not port_container:
        time.sleep(0.1)
    bokeh_port = port_container[0]


    col_plot = st.container()

    with col_plot:
        bokeh_url = f"http://localhost:{bokeh_port}/bkapp"
        st.markdown(
            f"""
            <div style="width:100%;">
            <iframe
                src="{bokeh_url}"
                style="display:block; width:100%; height:{iframe_height}px; border:none;"
            ></iframe>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.success("TMS + Hemispheres plotted successfully! 🚀")

# %%

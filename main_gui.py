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

    # ---- Bokeh server app ----
    def bkapp(doc):
        # sink function gets called by Bokeh callbacks on every range change
        def sink(hemi, start, end):
            with ranges_lock:
                ranges_store[hemi] = (start, end)

        data, tms_indexes = tms_utils.load_data(meta['input_file'])
        data_ds, factor = downsample(data, target_points=10000)

        plots = tms_utils.view_channels_bokeh_server(
            data_ds,
            meta["hemispheres"],
            tms_indexes // factor,
            fs=meta["sampling_rate"]/factor,
            range_sink=sink,   # << hook in sink
        )
        doc.add_root(bk_column(*plots, sizing_mode="stretch_width"))

    # Detect Streamlit port for websocket allowlist
    streamlit_port = os.environ.get("STREAMLIT_SERVER_PORT", "8501")

    # Start Bokeh server in background
    def run_bokeh_server(port_container):
        server = Server(
            {'/bkapp': bkapp},
            port=0,
            allow_websocket_origin=[f"localhost:{streamlit_port}"]
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

    # --- Two-column layout (plots left, blocks right) ---
    col_plot, col_right = st.columns([7, 3])

    num_channels = len(meta["hemispheres"])
    per_channel_px = 500
    iframe_height = per_channel_px * max(1, num_channels)
    iframe_width = 3000

    with col_plot:
        bokeh_url = f"http://localhost:{bokeh_port}/bkapp"
        components.html(
            f'''
            <iframe
                src="{bokeh_url}"
                style="width:100%; height:{iframe_height}px; border:none;"
            ></iframe>
            ''',
            height=iframe_height,
            width=iframe_width
)
    with col_right:
        render_text(
            'Experiment Blocks',
            font_color="black",
            font_weight="bold",          # normal or bold
            horizontal_alignment="center",  # left, center, right
            font_size=None,                # CSS font size (if None, use heading map)
            nowrap=True,                   # True = do not wrap, False = allow wrapping
            heading_level=3             # None for normal text, or 1–5 for heading style
        )

        with st.container(height=iframe_height, border=True):
            for part in meta['exp_structure']:
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    render_text(
                        part,
                        font_color="black",
                        font_weight="bold",          # normal or bold
                        horizontal_alignment="right",  # left, center, right
                        font_size=None,                # CSS font size (if None, use heading map)
                        nowrap=True,                   # True = do not wrap, False = allow wrapping
                        heading_level=5             # None for normal text, or 1–5 for heading style
                    )

                with c2:
                    st.text_input(label=part, label_visibility='collapsed', key=f"input_{part}")
                with c3:
                    st.button(label="Set", key=f"btn_{part}",
                              on_click=setRange, kwargs={'part': part})

    st.success("TMS + Hemispheres plotted successfully! 🚀")

# %%

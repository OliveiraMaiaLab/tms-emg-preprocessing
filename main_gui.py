# -*- coding: utf-8 -*-
"""
to execute on command line:
        
        streamlit run main_gui.py
"""


import os
import streamlit as st
import numpy as np
from bokeh.plotting import figure
from bokeh.models import HoverTool
import tmp_module as tms_utils

def downsample(data, target_points=2000):
    factor = max(1, data.shape[1] // target_points)
    return data[:, ::factor], factor

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
    st.session_state.step = "input"
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

    # Other fields
    meta["sampling_rate"] = st.number_input(
        "Sampling Rate (Hz)", value=meta["sampling_rate"], step=100
    )
    meta["subj_id"] = st.text_input("ID", value=meta["subj_id"])
    meta["session"] = st.number_input(
        "Session", min_value=1, value=meta["session"], step=1
    )

    # Hemisphere selection
    left = st.checkbox("Left hemisphere", value="left" in meta["hemispheres"] or not meta["hemispheres"])
    right = st.checkbox("Right hemisphere", value="right" in meta["hemispheres"])
    meta["hemispheres"] = []
    if left:
        meta["hemispheres"].append("left")
    if right:
        meta["hemispheres"].append("right")
    if st.button("Advance"):
        # Optionally, check if file exists
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
    hemi_str1 = " and ".join(meta["hemispheres"]) if meta["hemispheres"] else "none"
    hemi_str2 = 's' if 'and' in hemi_str1 else ''
    st.info(
        f"Using experiment template `{meta['template_file']}` to process file `{meta['input_file']}`.\n\n"
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
    st.title("TMS Signal Viewer 🎉")

    # --- Example: replace with actual EMG data loading ---
    # For demonstration, using random data: 3 channels, 10 sec at 4 kHz
    data, tms_indexes = tms_utils.load_data(meta['input_file'])

    data_ds, factor = downsample(data, target_points=10000)
    # Plot using your Bokeh function
    plots = tms_utils.view_channels_bokeh(data_ds, meta["hemispheres"], tms_indexes // factor, fs=meta["sampling_rate"]/factor)

    for p in plots:
        st.bokeh_chart(p, use_container_width=True)

    st.success("TMS + Hemispheres plotted successfully! 🚀")


# -*- coding: utf-8 -*-
"""
to activate conda env:
        conda activate cfom_mep_preprocessing

to execute on command line:
        
        streamlit run main_gui.py
"""


import os
import streamlit as st
import numpy as np
from bokeh.plotting import figure
from bokeh.models import HoverTool
import tmp_module as tms_utils
import json
import html

#%% Configure Layout  

st.set_page_config(layout="wide") 

#%% Style functions

def render_text(
    text,
    font_color="black",
    font_weight="normal",         # normal or bold
    horizontal_alignment="right", # left, center, right
    font_size="1.25rem",          # CSS font size
    nowrap=True                   # True = do not wrap, False = allow wrapping
):
    """
    Render styled text in Streamlit using HTML.

    Args:
        text (str): Text to display.
        font_color (str): Text color.
        font_weight (str): 'normal' or 'bold'.
        horizontal_alignment (str): 'left', 'center', 'right'.
        font_size (str): CSS font size (e.g., '1.25rem', '20px').
        nowrap (bool): If True, text will not wrap.
    """
    justify_map = {"left": "flex-start", "center": "center", "right": "flex-end"}
    h_align = justify_map.get(horizontal_alignment.lower(), "flex-end")
    
    white_space = "nowrap" if nowrap else "normal"

    html_code = (
        f"<div style='display: flex; width: 100%; justify-content: {h_align}; "
        f"font-weight: {font_weight}; color: {font_color}; font-size: {font_size}; "
        f"white-space: {white_space}; margin: 0;'>"
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

    def setRange():
        pass

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

    st.title("EMG Segmentation")

    # Arrange in columns
    col1, col2 = st.columns([7, 3])  # tweak ratios as needed

    # --- Example: replace with actual EMG data loading ---
    # For demonstration, using random data: 3 channels, 10 sec at 4 kHz
    data, tms_indexes = tms_utils.load_data(meta['input_file'])

    data_ds, factor = downsample(data, target_points=10000)
    # Plot using your Bokeh function
    with col1:
        plots = tms_utils.view_channels_bokeh(data_ds, meta["hemispheres"], tms_indexes // factor, fs=meta["sampling_rate"]/factor)

        for p in plots:
            st.bokeh_chart(p, use_container_width=True)

    with col2:
        render_text('Experiment Blocks',
                    font_size="30px",
                    font_color="black",
                    font_weight="bold",  # options: normal, bold
                    horizontal_alignment="center",  # left, center, right
                    nowrap =True )
        
        num_plots = len(plots)
        container_height = num_plots * 400  # px, estimate

        with st.container(height=container_height, border=True):  
            for part in meta['exp_structure']:
                row_col1, row_col2, row_col3 = st.columns([1, 1, 1])  # one row per experiment part
                with row_col1:
                    render_text(part,
                                font_size="20px",
                                font_color="black",
                                font_weight="bold",  # options: normal, bold
                                horizontal_alignment="right",  # left, center, right
                                nowrap =True )
                with row_col2:
                    st.text_input(
                        label=part,
                        label_visibility='collapsed',
                        key=f"input_{part}"
                    )
                with row_col3:
                    st.button(
                        label="Set",
                        key=f"btn_{part}",
                        on_click=setRange
                    )
    st.success("TMS + Hemispheres plotted successfully! 🚀")


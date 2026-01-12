# -*- coding: utf-8 -*-
"""
main_gui.py
-----------
Entry point and simple router for the multi-step Streamlit app.
Keeps only the minimal glue; each step manages its own UI + side effects.
"""
import os
import streamlit as st

from utils.persistence import load_persisted_defaults
from steps.step_input import run_step as step_input
from steps.step_confirmInputs import run_step as step_confirmInputs
from steps.step_segmentation import run_step as step_segmentation
from steps.step_mepWindow import run_step as step_mepWindow
from steps.step_peakChecking import run_step as step_peakCheck
from steps.step_peakCorrection import run_step as step_peakCorrection


st.set_page_config(layout="wide")

# -------------------------
# Bootstrap router state
# -------------------------
if "step" not in st.session_state:
    # st.session_state.step = "input"
    st.session_state.step = "peak_correction"

# -------------------------
# Bootstrap metadata
# -------------------------
if "metadata" not in st.session_state:
    # load_persisted_defaults now returns a dict (auto-upgraded)
    defaults = load_persisted_defaults()

    st.session_state.metadata = {
        "template_file": defaults.get("template_file", ""),
        "input_file": defaults.get("input_file", ""),
        "output_dir": defaults.get("output_dir", ""),

        # NEW persisted fields
        "data_dir": defaults.get("data_dir", ""),
        "researcher_id": defaults.get("researcher_id", ""),
        "version": defaults.get("version", ""),

        "sampling_rate": 4000,
        "subj_id": "example_sub",
        "session": 1,
        "hemispheres": ["left"],
        "_script_dir": os.path.dirname(os.path.abspath(__file__)),
    }

meta = st.session_state.metadata

# -------------------------
# Route map
# -------------------------
ROUTES = {
    "input": step_input,
    "confirmInputs": step_confirmInputs,
    "segmentation": step_segmentation,
    "mep_window": step_mepWindow,
    "peak_checking": step_peakCheck,
    "peak_correction": step_peakCorrection,
}

ROUTES.get(st.session_state.step, step_input)(meta)

   
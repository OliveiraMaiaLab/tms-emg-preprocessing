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

st.set_page_config(layout="wide")

# bootstrap state
if "step" not in st.session_state:
    # st.session_state.step = "input"
    st.session_state.step = "segmentation"

if "metadata" not in st.session_state:
    # ✨ load_persisted_defaults now returns (template_file, input_file, output_dir)
    tdef, idef, odef = load_persisted_defaults()
    st.session_state.metadata = {
        "template_file": tdef,
        "input_file": idef,
        "output_dir": odef,
        "sampling_rate": 4000,
        "subj_id": "example_sub",
        "session": 1,
        "hemispheres": ["left"],
        "_script_dir": os.path.dirname(os.path.abspath(__file__)),
    }

meta = st.session_state.metadata

# route
ROUTES = {
    "input": step_input,
    "confirmInputs": step_confirmInputs,
    "segmentation": step_segmentation,
    "mep_window": step_mepWindow,
}
ROUTES.get(st.session_state.step, step_input)(meta)

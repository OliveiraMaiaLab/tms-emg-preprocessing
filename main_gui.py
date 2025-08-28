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

st.set_page_config(layout="wide")

# bootstrap state
if "step" not in st.session_state:
    st.session_state.step = "input"

if "metadata" not in st.session_state:
    tdef, idef = load_persisted_defaults()
    st.session_state.metadata = {
        "template_file": tdef,
        "input_file": idef,
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
    "confirm": step_confirmInputs,
    "proceed": step_segmentation,
}
ROUTES.get(st.session_state.step, step_input)(meta)

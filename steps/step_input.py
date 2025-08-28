"""
step_input.py
-------------
Step 1: Collect paths and general metadata. Validates inputs and saves
last-used paths to settings.
"""
import os, numpy as np
import streamlit as st
from utils.persistence import load_persisted_defaults, save_persisted_defaults

def run_step(meta: dict):
    st.title("TMS-EMG Preprocessing GUI")

    if meta.get("_first_visit_input", True):
        tdef, idef = load_persisted_defaults()
        meta.setdefault("template_file", tdef)
        meta.setdefault("input_file", idef)
        meta["_first_visit_input"] = False

    meta["template_file"] = st.text_input("Select experiment template file", value=meta["template_file"])
    meta["input_file"] = st.text_input("Select EMG file", value=meta["input_file"])

    col1, col2, col3, col4, _ = st.columns([2,1,1,1,3])
    with col1: meta["subj_id"] = st.text_input("ID", value=meta.get("subj_id","example_sub"))
    with col2: meta["session"] = st.number_input("Session", min_value=1, value=int(meta.get("session",1)), step=1)
    with col3: meta["sampling_rate"] = st.number_input("Sampling Rate (Hz)", value=int(meta.get("sampling_rate",4000)), step=100)
    with col4:
        st.write("Hemisphere(s):")
        c1, c2 = st.columns(2)
        left  = c1.checkbox("Left",  value=("left"  in meta.get("hemispheres", ["left"])))
        right = c2.checkbox("Right", value=("right" in meta.get("hemispheres", [])))
        meta["hemispheres"] = [h for h,v in (("left",left), ("right",right)) if v]

    if st.button("Advance"):
        if not os.path.exists(meta["template_file"]):
            st.error("Template file does not exist!")
        elif not os.path.exists(meta["input_file"]):
            st.error("EMG file does not exist!")
        elif not meta["hemispheres"]:
            st.error("Select at least one hemisphere!")
        else:
            save_persisted_defaults(meta["template_file"], meta["input_file"])
            st.session_state.step = "confirm"
            st.rerun()

"""
step_segmentation.py
---------------
Step 3: Start the Bokeh app and embed it. The right panel's 'Save' writes
ranges into the subject/session JSON under 'segmentation'.
"""
import threading
import streamlit as st
from utils.bokeh_embed import start_bokeh_app

def run_step(meta: dict):
    if "_bokeh_port" not in st.session_state:
        st.session_state["_ranges_store"] = {}
        st.session_state["_ranges_lock"] = threading.Lock()
        port = start_bokeh_app(
            meta=meta,
            session_file=st.session_state["_session_file"],
            exp_structure=meta["exp_structure"],
            SCRIPT_DIR=meta.get("_script_dir","."),
            ranges_store=st.session_state["_ranges_store"],
            ranges_lock=st.session_state["_ranges_lock"],
        )
        st.session_state["_bokeh_port"] = port

    iframe_height = 800 * max(1, len(meta["hemispheres"]))
    bokeh_url = f"http://localhost:{st.session_state['_bokeh_port']}/bkapp"
    st.markdown(
        f"<div style='width:100%;'><iframe src='{bokeh_url}' "
        f"style='display:block;width:100%;height:{iframe_height}px;border:none;'></iframe></div>",
        unsafe_allow_html=True
    )
    st.success("TMS + Hemispheres plotted successfully! 🚀")

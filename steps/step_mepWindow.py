"""
step_mepWindow.py
-----------------
Step 4: Define window to compute MEP amplitude
"""
import threading
import streamlit as st
import streamlit.components.v1 as components

from utils.persistence import (
    ensure_metadata,
    ensure_template_loaded,
    ensure_session_file,
)
from utils.layout import render_text, step_nav

# Import the Bokeh server launcher
from utils.bk_mepoverlap_embedding import start_bokeh_app


def run_step(meta: dict):
    meta = ensure_metadata()
    meta = ensure_template_loaded(meta)
    session_file = ensure_session_file(meta)

    step_nav(
        "mep_window",
        back_step="segmentation",
        next_step="Peak checking",
        disabled_next=True,  # keep your current behavior for now
    )

    st.title("MEP window")
    render_text(
        "MEP window",
        font_color="black",
        font_weight="normal",
        horizontal_alignment="center",
        font_size=None,
        nowrap=True,
        heading_level=1,
    )

    # -------------------- BOKEH PLOT BELOW --------------------

    # --- Ephemeral runtime state (shared with the Bokeh thread)
    if "_ranges_store" not in st.session_state or "_ranges_lock" not in st.session_state:
        st.session_state["_ranges_store"] = {}
        st.session_state["_ranges_lock"] = threading.Lock()

    # Restart Bokeh if context changes (prevents “stale plot” across subjects/sessions)
    bokeh_key = (
        session_file,
        meta.get("input_file"),
        meta.get("sampling_rate"),
    )

    if st.session_state.get("_bokeh_key") != bokeh_key:
        st.session_state["_bokeh_key"] = bokeh_key
        st.session_state["_bokeh_port"] = start_bokeh_app(
            meta=meta,
            session_file=session_file,
            exp_structure=meta.get("exp_structure", []),
            SCRIPT_DIR=meta.get("_script_dir", "."),
            ranges_store=st.session_state["_ranges_store"],
            ranges_lock=st.session_state["_ranges_lock"],
        )

    # Embed the Bokeh app
    components.iframe(
        src=f"http://127.0.0.1:{st.session_state['_bokeh_port']}/bkapp",
        height=540,
        scrolling=True,
    )

    # Optional: show current selected window live (debug)
    with st.session_state["_ranges_lock"]:
        epoch_window = st.session_state["_ranges_store"].get("epoch_window")
    st.caption(f"Current epoch window (s): {epoch_window}")
"""
step_peakChecking.py
-----------------
Step 4: Define window to compute MEP amplitude
"""
import threading
import streamlit as st

from utils.persistence import (
    ensure_metadata,
    ensure_template_loaded,
    ensure_session_file,
)
from utils.layout import render_text, step_nav
import json
from pathlib import Path

# Import the Bokeh server launcher
from utils.bk_mepOverlap_embedding import start_bokeh_app



def run_step(meta: dict):
    meta = ensure_metadata()
    meta = ensure_template_loaded(meta)
    session_file = ensure_session_file(meta)

    step_nav(
        "peak_checking",
        back_step="mep_window",
        disabled_next=True,
    )

    # st.title("MEP window")
    render_text(
        "Peak checking dev",
        font_color="black",
        font_weight="normal",
        horizontal_alignment="center",
        font_size=None,
        nowrap=True,
        heading_level=1,
    )

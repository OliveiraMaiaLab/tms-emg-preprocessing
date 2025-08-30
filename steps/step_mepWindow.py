"""
step_mepWindow.py
---------------
Step 4: Define window to compute MEP amplitude
"""
import json, numpy as np
import streamlit as st
from utils.persistence import (
    ensure_metadata,
    ensure_template_loaded,
    ensure_session_file,
)
from utils.layout import render_text, step_nav

def run_step(meta: dict):
    meta = ensure_metadata()
    meta = ensure_template_loaded(meta)
    session_file = ensure_session_file(meta)

    step_nav("mep_window", 
             back_step="segmentation", 
             next_step = 'Peak checking',
             disabled_next=True)

    
    st.title("MEP window")
    render_text('MEP window', font_color="black", font_weight="normal",
                    horizontal_alignment="center", font_size=None, nowrap=True, heading_level=1)
    
    
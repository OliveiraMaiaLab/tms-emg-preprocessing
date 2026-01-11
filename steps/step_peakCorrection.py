
from __future__ import annotations

from pathlib import Path
import streamlit as st

from utils.layout import render_text, step_nav

PREV_STEP = "peak_checking"
THIS_STEP = "peak_correction"
NEXT_STEP = None

from utils.persistence import (
    ensure_metadata,
    ensure_template_loaded,
    ensure_session_file,
)

def run_step(meta: dict):
    meta = ensure_metadata()
    meta = ensure_template_loaded(meta)
    session_file = Path(ensure_session_file(meta))



    step_nav(
        THIS_STEP,
        step_title = "Peak Correction",
        next_step=NEXT_STEP,
        back_step=PREV_STEP,
        disabled_next=True)

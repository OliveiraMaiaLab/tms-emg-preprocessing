# app/main_gui.py
# -*- coding: utf-8 -*-
"""
main_gui.py
-----------
Entry point and simple router for the multi-step Streamlit app.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- Make repo root importable so `import app...` works ---
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from app.utils.persistence import load_persisted_defaults
from app.steps.step_input import run_step as step_input
from app.steps.step_confirmInputs import run_step as step_confirmInputs
from app.steps.step_segmentation import run_step as step_segmentation
from app.steps.step_mepWindow import run_step as step_mepWindow
from app.steps.step_peakChecking import run_step as step_peakCheck
from app.steps.step_peakCorrection import run_step as step_peakCorrection

st.set_page_config(layout="wide")


# -------------------------
# Bootstrap router state
# -------------------------
if "step" not in st.session_state:
    st.session_state.step = "input"


# -------------------------
# Bootstrap metadata
# -------------------------
if "metadata" not in st.session_state:
    defaults = load_persisted_defaults()

    script_dir = Path(__file__).resolve().parent  # .../repo/app

    st.session_state.metadata = {
        "template_file": defaults.get("template_file", ""),
        "input_file": defaults.get("input_file", ""),
        "output_dir": defaults.get("output_dir", ""),

        "data_dir": defaults.get("data_dir", ""),
        "researcher_id": defaults.get("researcher_id", ""),
        "version": defaults.get("version", ""),

        "sampling_rate": int(defaults.get("sampling_rate", 4000) or 4000),
        "subj_id": defaults.get("subj_id", "example_sub") or "example_sub",
        "session": int(defaults.get("session", 1) or 1),
        "hemispheres": defaults.get("hemispheres", ["left"]) or ["left"],

        # used by embedded apps / dash reload signatures / etc.
        "_script_dir": str(script_dir),
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

ROUTES.get(st.session_state.step, step_input)

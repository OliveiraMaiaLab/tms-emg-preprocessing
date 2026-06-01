# app/main_gui.py
# -*- coding: utf-8 -*-
"""
Entry point and router for the multi-step Streamlit app.
"""

from __future__ import annotations

import sys
from pathlib import Path
import importlib

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

st.set_page_config(layout="wide")


def _load_step_runner(module_path: str, func_name: str = "run_step"):
    mod = importlib.import_module(module_path)
    fn  = getattr(mod, func_name, None)
    if fn is None:
        raise AttributeError(f"{module_path} has no function '{func_name}'")
    return fn


if "step" not in st.session_state:
    st.session_state.step = "input"

if "metadata" not in st.session_state:
    persistence = importlib.import_module("app.utils.persistence")
    defaults    = persistence.load_persisted_defaults()
    script_dir  = Path(__file__).resolve().parent

    st.session_state.metadata = {
        "template_file":  defaults.get("template_file", ""),
        "input_file":     defaults.get("input_file", ""),
        "output_dir":     defaults.get("output_dir", ""),
        "data_dir":       defaults.get("data_dir", ""),
        "researcher_id":  defaults.get("researcher_id", ""),
        "version":        defaults.get("version", ""),
        "sampling_rate":  int(defaults.get("sampling_rate", 4000) or 4000),
        "subj_id":        defaults.get("subj_id", "example_sub") or "example_sub",
        "session":        int(defaults.get("session", 1) or 1),
        "hemispheres":    defaults.get("hemispheres", ["left"]) or ["left"],
        "_script_dir":    str(script_dir),
    }

meta = st.session_state.metadata

ROUTES = {
    "input":           "app.steps.step_input",
    "confirmInputs":   "app.steps.step_confirmInputs",
    "segmentation":    "app.steps.step_segmentation",
    "mep_window":      "app.steps.step_mepWindow",
    "peak_checking":   "app.steps.step_peakChecking",
    "peak_correction": "app.steps.step_peakCorrection",
    "review_flag":     "app.steps.step_reviewFlag",   # new
}

step_name   = st.session_state.step
module_path = ROUTES.get(step_name)

if module_path is None:
    st.warning(
        f"Unknown step '{step_name}' — resetting to input. "
        "This is likely a bug; please report it."
    )
    st.session_state.step = "input"
    module_path = ROUTES["input"]

run_step = _load_step_runner(module_path)
run_step(meta)

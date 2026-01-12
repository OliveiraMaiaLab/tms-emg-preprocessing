"""
steps/step_input.py
-------------------
Step 1: Collect paths and general metadata. Validates inputs and saves
last-used paths to settings.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from utils.persistence import (
    ensure_metadata,
    ensure_output_dir,
    save_persisted_defaults,
)


def run_step(meta: dict):
    meta = ensure_metadata()

    # Flash message (from finishing other steps)
    msg = st.session_state.pop("_global_flash_success", None)
    if msg:
        st.success(msg)

    st.title("TMS-EMG Preprocessing GUI")

    # -------------------------
    # Inputs (persisted defaults already loaded by ensure_metadata)
    # -------------------------
    meta["template_file"] = st.text_input(
        "Experiment template file",
        value=str(meta.get("template_file", "")),
    )

    meta["data_dir"] = st.text_input(
        "Data directory (folder containing EMG files)",
        value=str(meta.get("data_dir", "")),
    )

    meta["input_file"] = st.text_input(
        "EMG file",
        value=str(meta.get("input_file", "")),
    )

    meta["output_dir"] = st.text_input(
        "Output folder (where to save session files and results)",
        value=str(meta.get("output_dir", os.getcwd())),
    )

    meta["researcher_id"] = st.text_input(
        "Researcher ID",
        value=str(meta.get("researcher_id", "")),
    )

    # pipeline version is informative here; keep in meta (and persisted by save)
    st.caption(f"Pipeline version: `{meta.get('version', '')}`")

    # -------------------------
    # Subject/session settings
    # -------------------------
    col1, col2, col3, col4, _ = st.columns([2, 1, 1, 1, 1])
    with col1:
        meta["subj_id"] = st.text_input("Subject ID", value=str(meta.get("subj_id", "example_sub")))
    with col2:
        meta["session"] = st.number_input("Session", min_value=1, value=int(meta.get("session", 1)), step=1)
    with col3:
        meta["sampling_rate"] = st.number_input(
            "Sampling Rate (Hz)", value=int(meta.get("sampling_rate", 4000)), step=100
        )
    with col4:
        st.markdown("Hemisphere(s):")
        c1, c2 = st.columns(2)
        left = c1.checkbox("Left", value=("left" in meta.get("hemispheres", ["left"])))
        right = c2.checkbox("Right", value=("right" in meta.get("hemispheres", [])))
        meta["hemispheres"] = [h for h, v in (("left", left), ("right", right)) if v]

    # -------------------------
    # Advance
    # -------------------------
    if st.button("Advance ▶", type="primary"):
        # Basic validation
        if not meta.get("template_file") or not os.path.exists(meta["template_file"]):
            st.error("Template file does not exist.")
            st.stop()

        if not meta.get("input_file") or not os.path.exists(meta["input_file"]):
            st.error("EMG file does not exist.")
            st.stop()

        if not meta.get("hemispheres"):
            st.error("Select at least one hemisphere.")
            st.stop()

        # Make output dir (and show toast)
        try:
            meta["output_dir"] = ensure_output_dir(meta.get("output_dir", ""), show_toast=True)
        except Exception:
            st.stop()

        # Persist GUI defaults (includes data_dir, researcher_id, version)
        save_persisted_defaults(
            template_path=meta["template_file"],
            input_path=meta["input_file"],
            output_dir=meta["output_dir"],
            data_dir=meta.get("data_dir", ""),
            researcher_id=meta.get("researcher_id", ""),
        )

        st.session_state.step = "confirmInputs"
        st.rerun()

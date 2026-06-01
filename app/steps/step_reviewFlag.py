# app/steps/step_reviewFlag.py
# -*- coding: utf-8 -*-
"""
steps/step_reviewFlag.py
------------------------
Final step: ask the researcher whether this file needs further checking
before being included in analysis.

Writes to session JSON:
  "flag_for_review": true | false
  "review_note":     "" | "<free text>"

Then updates the processed sessions registry and returns to the input step.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.utils.persistence import (
    ensure_metadata,
    ensure_template_loaded,
    ensure_session_file,
    update_processed_sessions_registry,
    write_review_flag,
)
from app.utils.tms_module import read_json

PREV_STEP = "peak_correction"
THIS_STEP = "review_flag"


def _do_finish(
    meta: dict,
    session_file: Path,
    flag_for_review: bool,
    review_note: str,
) -> None:
    """Write flag + note, update registry, reset session state, go to input."""
    write_review_flag(
        session_file=session_file,
        flag_for_review=flag_for_review,
        review_note=review_note,
    )
    update_processed_sessions_registry(
        data_file=meta.get("input_file", "UNKNOWN"),
        session_file=session_file,
        researcher_id=meta.get("researcher_id", ""),
        pipeline_version=meta.get("version", ""),
    )

    label = session_file.name
    if flag_for_review:
        flash = f'"{label}" saved and **flagged for review**.'
    else:
        flash = f'"{label}" saved successfully ✅'

    # Clear pipeline-specific session state so the next file starts clean.
    for key in list(st.session_state.keys()):
        if key not in ("metadata", "step"):
            del st.session_state[key]

    st.session_state["_global_flash_success"] = flash
    st.session_state.step = "input"
    st.rerun()


def run_step(meta: dict):
    meta         = ensure_metadata()
    meta         = ensure_template_loaded(meta)
    session_file = Path(ensure_session_file(meta))

    # Read any previously saved values so the UI reflects them on re-visit.
    session          = read_json(session_file)
    saved_flag       = bool(session.get("flag_for_review", False))
    saved_note       = str(session.get("review_note", "") or "")

    # ---- back button ----
    if st.button("◀ Back to peak correction"):
        st.session_state.step = PREV_STEP
        st.rerun()

    st.divider()

    # ---- centred card layout ----
    _, centre, _ = st.columns([1, 2, 1])

    with centre:
        st.markdown("## Final check")
        st.markdown(
            f"You have finished preprocessing **{session_file.name}**.  \n"
            "Before saving, please indicate whether this file should be "
            "flagged for further review before being included in the analysis."
        )

        st.markdown("---")

        flag = st.radio(
            "Flag this file for review?",
            options=["No — include in analysis as-is", "Yes — flag for review"],
            index=1 if saved_flag else 0,
            key="_review_flag_radio",
            horizontal=True,
        )
        flag_for_review = flag.startswith("Yes")

        # Note field — always visible, but labelled to make clear it is optional.
        note_placeholder = (
            "Optional: briefly describe what needs checking "
            "(e.g. noisy baseline in bmeps block, artifact around pulse 42…)"
        )
        note = st.text_area(
            "Review note (optional)",
            value=saved_note,
            placeholder=note_placeholder,
            height=100,
            key="_review_note_input",
        )

        st.markdown("---")

        # ---- finish buttons ----
        # Two explicit buttons so the action is unambiguous.
        if flag_for_review:
            btn_label = "💾 Save & flag for review"
            btn_type  = "secondary"
            help_text = "The file will be saved and marked as needing further inspection."
        else:
            btn_label = "✅ Save & finish"
            btn_type  = "primary"
            help_text = "The file will be saved and marked as ready for analysis."

        if st.button(btn_label, type=btn_type, use_container_width=True, help=help_text):
            _do_finish(
                meta=meta,
                session_file=session_file,
                flag_for_review=flag_for_review,
                review_note=note.strip(),
            )

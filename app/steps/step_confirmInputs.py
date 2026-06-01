# app/steps/step_confirmInputs.py
# -*- coding: utf-8 -*-
"""
steps/step_confirmInputs.py
---------------
Step 2: Show a summary; on confirm, create/update the subject/session JSON.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.utils.persistence import (
    create_or_update_session_file,
    ensure_metadata,
    ensure_template_loaded,
)


def run_step(meta: dict):
    meta = ensure_metadata()
    # Bug #1: ensure_template_loaded already resolves the template path via
    # resolve_template_path(), sets meta["exp_name"], meta["channels"] (as list),
    # and meta["exp_structure"] (including emg_ref logic). The previous code
    # re-opened the file with open(meta["template_file"]) which failed when
    # template_file was a bare filename rather than a full path, and also
    # duplicated — and diverged from — the logic in ensure_template_loaded.
    meta = ensure_template_loaded(meta)

    st.subheader("Please confirm your choices:")

    hemi_str = " and ".join(meta["hemispheres"]) if meta["hemispheres"] else "none"
    input_name = Path(meta["input_file"]).name

    st.markdown(
        """
        <style>
        .info-box{
            max-width: 560px;
            margin: 0 auto 1rem;
            padding: .9rem 1.2rem;
            background: rgba(43,144,217,.08);
            border-radius: .25rem;
            text-align: center;
            line-height: 1.5;
        }
        .info-box .kv{ font-size:1.15em; font-weight:600; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="info-box">
        Processing file <span class="kv">{input_name}</span> of
        <span class="kv">{meta['exp_name']}</span> experiment.
        Subject <span class="kv">{meta['subj_id']}</span>, session
        <span class="kv">{meta['session']}</span>, hemisphere(s):
        <span class="kv">{hemi_str}</span>.
        <br><br>
        <span class="kv">Proceed?</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, c1, _, c2, _ = st.columns([2, 1, 1, 1, 2])
    with c1:
        if st.button("No, go back", use_container_width=True):
            st.session_state.step = "input"
            st.rerun()
    with c2:
        if st.button("Yes", use_container_width=True):
            session_file_path = create_or_update_session_file(meta, meta["exp_structure"])
            st.session_state["_session_file"] = session_file_path
            st.session_state.step = "segmentation"
            st.rerun()

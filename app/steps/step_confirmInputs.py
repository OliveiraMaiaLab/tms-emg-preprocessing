# app/steps/step_confirmInputs.py
# -*- coding: utf-8 -*-
"""
steps/step_confirmInputs.py
---------------
Step 2: Show a summary; on confirm, create/update the subject/session JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st

from app.utils.persistence import (
    create_or_update_session_file,
    ensure_metadata,
    ensure_template_loaded,
)


def run_step(meta: dict):
    meta = ensure_metadata()
    meta = ensure_template_loaded(meta)

    st.subheader("Please confirm your choices:")

    with open(meta["template_file"], "r") as f:
        template = json.load(f)

    meta["exp_name"] = template["experiment_name"]
    meta["channels"] = np.array(
        [
            template["channels"]["synch_pulse"],
            template["channels"]["right"],
            template["channels"]["left"],
        ]
    )
    meta["exp_structure"] = template["experiment_structure"]

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

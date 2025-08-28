"""
step_confirm.py
---------------
Step 2: Show a summary; on confirm, create/update the subject/session JSON.
"""
import json, numpy as np
import streamlit as st
from utils.persistence import create_or_update_session_file

def run_step(meta: dict):
    st.subheader("Please confirm your choices:")

    with open(meta["template_file"], "r") as f:
        template = json.load(f)
    meta['exp_name'] = template["experiment_name"]
    meta['channels'] = np.array([
        template["channels"]["synch_pulse"],
        template["channels"]["right"],
        template["channels"]["left"]
    ])
    meta['exp_structure'] = template["experiment_structure"]

    hemi_str = " and ".join(meta["hemispheres"]) if meta["hemispheres"] else "none"
    st.info(
        f"Process file `{meta['input_file']}` of `{meta['exp_name']}` experiment.\n\n"
        f"Subject `{meta['subj_id']}`, session `{meta['session']}`, hemisphere(s): **{hemi_str}**.\n\nProceed?"
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("No, go back"):
            st.session_state.step = "input"; st.rerun()
    with c2:
        if st.button("Yes, proceed"):
            session_file_path = create_or_update_session_file(meta, meta['exp_structure'])
            st.session_state["_session_file"] = session_file_path
            st.session_state.step = "proceed"
            st.rerun()

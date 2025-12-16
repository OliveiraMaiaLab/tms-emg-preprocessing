"""
step_mepWindow.py
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

# Import the Bokeh server launcher
from utils.bk_mepOverlap_embedding import start_bokeh_app


def run_step(meta: dict):
    meta = ensure_metadata()
    meta = ensure_template_loaded(meta)
    session_file = ensure_session_file(meta)

    step_nav(
        "mep_window",
        back_step="segmentation",
        next_step="Peak checking",
        disabled_next=True,
    )

    st.title("MEP window")
    render_text(
        "MEP window",
        font_color="black",
        font_weight="normal",
        horizontal_alignment="center",
        font_size=None,
        nowrap=True,
        heading_level=1,
    )

    # -------------------- BOKEH PLOT BELOW --------------------

    # --- Ephemeral runtime state (shared with the Bokeh thread)
    if "_ranges_store" not in st.session_state or "_ranges_lock" not in st.session_state:
        st.session_state["_ranges_store"] = {}
        st.session_state["_ranges_lock"] = threading.Lock()

    # Restart Bokeh if context changes
    bokeh_key = (
        session_file,
        meta.get("input_file"),
        meta.get("sampling_rate"),
    )

    if st.session_state.get("_bokeh_key") != bokeh_key:
        st.session_state["_bokeh_key"] = bokeh_key
        st.session_state["_bokeh_port"] = start_bokeh_app(
            meta=meta,
            session_file=session_file,
            exp_structure=meta.get("exp_structure", []),
            SCRIPT_DIR=meta.get("_script_dir", "."),
            ranges_store=st.session_state["_ranges_store"],
            ranges_lock=st.session_state["_ranges_lock"],
        )

    # ---- Responsive, centered, LARGE iframe ----
    ASPECT_W = 16
    ASPECT_H = 9

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewBlockContainer"] {{
            max-width: 100% !important;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
        }}

        section.main > div {{
            max-width: 100% !important;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
        }}

        html, body, [data-testid="stAppViewContainer"] {{
            overflow-x: hidden;
        }}

        .bk-viewport-band {{
            width: 100vw;
            position: relative;
            left: 50%;
            margin-left: -50vw;

            display: flex;
            justify-content: center;
            align-items: center;
        }}

        /* Height-driven sizing (fills vertical space), width computed from aspect ratio */
        .bk-viewport-band iframe {{
            display: block;
            border: none;

            height: 92vh;  /* <-- main knob: increase/decrease */

            width: calc(92vh * ({ASPECT_W} / {ASPECT_H}));
            max-width: 100vw;   /* don't overflow horizontally */
            max-height: 92vh;   /* keep consistent */
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    bokeh_url = f"http://localhost:{st.session_state['_bokeh_port']}/bkapp"

    st.markdown(
        f"""
        <div class="bk-viewport-band">
            <iframe src="{bokeh_url}" scrolling="no"></iframe>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Optional: live debug readout
    with st.session_state["_ranges_lock"]:
        epoch_window = st.session_state["_ranges_store"].get("epoch_window")
    st.caption(f"Current epoch window (s): {epoch_window}")


"""
step_mepWindow.py
-----------------
Step 4: Define window to compute MEP amplitude
"""
import threading
import streamlit as st
import json
from pathlib import Path

from utils.persistence import (
    ensure_metadata,
    ensure_template_loaded,
    ensure_session_file,
)
from utils.layout import render_text, step_nav
from utils.bk_mepOverlap_embedding import start_bokeh_app


def save_mep_window_to_session(session_file: str, window_s: tuple[float, float]) -> None:
    """
    Persist the selected MEP window (relative to pulse, in seconds).

    Writes:
        meps.window = [beg, end]
    """
    beg, end = map(float, window_s)
    if beg >= end:
        raise ValueError(f"Invalid MEP window: {beg} >= {end}")

    try:
        js = json.loads(Path(session_file).read_text())
    except Exception:
        js = {}

    meps = js.get("meps") or {}
    meps["window"] = [beg, end]
    js["meps"] = meps

    Path(session_file).write_text(json.dumps(js, indent=2))


def run_step(meta: dict):
    meta = ensure_metadata()
    meta = ensure_template_loaded(meta)
    session_file = ensure_session_file(meta)

    # --- Ephemeral runtime state (must exist BEFORE step_nav on_next runs)
    if "_ranges_store" not in st.session_state:
        st.session_state["_ranges_store"] = {}
    if "_ranges_lock" not in st.session_state:
        st.session_state["_ranges_lock"] = threading.Lock()

    # on_next callback (save only; step_nav will navigate)
    def _on_next():
        with st.session_state["_ranges_lock"]:
            win = st.session_state["_ranges_store"].get("epoch_window")

        if not win:
            st.toast("Select a window before advancing", icon="⚠️")
            return False

        save_mep_window_to_session(session_file, win)
        return True


    step_nav(
        "mep_window",
        back_step="segmentation",
        next_step="peak_checking",
        on_next=_on_next,
        disabled_next=False,
    )  
    render_text(
        "MEP window definition",
        font_color="black",
        font_weight="normal",
        horizontal_alignment="center",
        font_size=None,
        nowrap=True,
        heading_level=1,
    )

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

    # --- iframe styling
    st.markdown(
        """
        <style>
        [data-testid="stAppViewBlockContainer"] {
            max-width: 100% !important;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
        }

        section.main > div {
            max-width: 100% !important;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
        }

        html, body, [data-testid="stAppViewContainer"] {
            overflow-x: hidden;
        }

        .bk-viewport-band {
            width: 92vw;
            position: relative;
            left: 50%;
            margin-left: -46vw;

            display: flex;
            justify-content: center;
        }

        .bk-viewport-band iframe {
            display: block;
            border: none;

            width: 92vw;
            height: 46vw;     /* 2:1 ratio */
            max-height: 65vh; /* keeps buttons visible */
        }
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

    # debug readout
    with st.session_state["_ranges_lock"]:
        epoch_window = st.session_state["_ranges_store"].get("epoch_window")
    st.caption(f"Current epoch window (s): {epoch_window}")

"""
step_segmentation.py
--------------------
Embeds the Bokeh viewer and writes selected ranges into the subject/session JSON.
Defensive: hydrates missing state so deep-linking works.
"""
import threading
import streamlit as st
from utils.persistence import (
    ensure_metadata,
    ensure_template_loaded,
    ensure_session_file,
    is_segmentation_complete,
)
from utils.bk_segmentation_embedding import start_bokeh_app
from utils.layout import render_text, step_nav
import json
from pathlib import Path


def _segmentation_missing_flat(session_file: str, parts: list[str]) -> list[str]:
    """Return the list of parts that do NOT have a valid first [start, end] in the flat schema."""
    try:
        data = json.loads(Path(session_file).read_text())
    except Exception:
        return parts[:]  # if file can't be read, treat all as missing
    seg = data.get("segmentation", {}) or {}

    def ok(v):
        if not (isinstance(v, list) and v):
            return False
        first = v[0]
        if not (isinstance(first, (list, tuple)) and len(first) >= 2):
            return False
        try:
            s, e = float(first[0]), float(first[1])
        except (TypeError, ValueError):
            return False
        return s < e

    return [p for p in parts if not ok(seg.get(p, []))]


def _is_segmentation_complete_flat(session_file: str, parts: list[str]) -> bool:
    """Return True if every part in `parts` has a valid first [start, end] in the flat schema."""
    try:
        data = json.loads(Path(session_file).read_text())
    except Exception:
        return False
    seg = data.get("segmentation", {}) or {}

    def ok(v):
        if not (isinstance(v, list) and v):
            return False
        first = v[0]
        if not (isinstance(first, (list, tuple)) and len(first) >= 2):
            return False
        try:
            s, e = float(first[0]), float(first[1])
        except (TypeError, ValueError):
            return False
        return s < e

    return all(ok(seg.get(p, [])) for p in parts)


NEXT_AFTER_SEGMENTATION = "mep_window"

def run_step(meta: dict):
    meta = ensure_metadata()
    meta = ensure_template_loaded(meta)
    session_file = ensure_session_file(meta)

    # --- Nav bar with Back and smart Advance
    def _try_advance():
        required_parts = list(meta.get("exp_structure", []))
        missing = _segmentation_missing_flat(session_file, required_parts)

        if not missing:
            if NEXT_AFTER_SEGMENTATION:
                st.session_state.step = NEXT_AFTER_SEGMENTATION
            else:
                # show later in main render
                st.session_state["_segmentation_note_no_next"] = True
        else:
            # mark for display in the main render (callbacks can't render UI)
            st.session_state["_segmentation_incomplete"] = True
            st.session_state["_segmentation_missing"] = missing


    step_nav(
        "segmentation",
        back_step="confirm",
        next_step=NEXT_AFTER_SEGMENTATION,
        on_next=_try_advance,
        right_label="Advance ▶",
    )

    # Show any messages requested by the callback
    if st.session_state.pop("_segmentation_incomplete", False):
        missing = st.session_state.pop("_segmentation_missing", [])
        msg = "Finish segmentation before advancing"
        if missing:
            msg += f": {', '.join(missing)}"
        try:
            st.toast(msg, icon="⚠️")
        except Exception:
            st.warning(msg)

    if st.session_state.pop("_segmentation_note_no_next", False):
        try:
            st.toast("Segmentation complete — no next step configured.", icon="ℹ️")
        except Exception:
            st.info("Segmentation complete — no next step configured.")

    # --- Ephemeral runtime state
    if "_ranges_store" not in st.session_state:
        st.session_state["_ranges_store"] = {}
    if "_ranges_lock" not in st.session_state:
        st.session_state["_ranges_lock"] = threading.Lock()

    if "_bokeh_port" not in st.session_state:
        st.session_state["_bokeh_port"] = start_bokeh_app(
            meta=meta,
            session_file=session_file,
            exp_structure=meta["exp_structure"],
            SCRIPT_DIR=meta.get("_script_dir", "."),
            ranges_store=st.session_state["_ranges_store"],
            ranges_lock=st.session_state["_ranges_lock"],
        )


    render_text('EMG Segmentation', font_color="black", font_weight="bold",
                horizontal_alignment="center", font_size=None, nowrap=True, heading_level=1)

    # Hide horizontal overflow globally + in the iframe wrapper
    st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] { overflow-x: hidden; }
    .bk-iframe-wrap { width: 100%; overflow-x: hidden; }
    .bk-iframe-wrap iframe { display: block; width: 100%; border: none; }
    </style>
    """, unsafe_allow_html=True)

    iframe_height = 550 * max(1, len(meta["hemispheres"]))
    bokeh_url = f"http://localhost:{st.session_state['_bokeh_port']}/bkapp"

    st.markdown(
        f"""
        <div class="bk-iframe-wrap">
        <iframe src="{bokeh_url}" height="{iframe_height}" scrolling="no"></iframe>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.caption(f"Session file: `{session_file}`")
"""
steps/step_mepWindow.py
-----------------
Step 4: Define window to compute MEP amplitude (relative to pulse).

Writes:
  session["mep_window"] = [beg_s, end_s]   # seconds (relative to pulse)

Then computes per-block min/max within that window:
  session["meps"][block][hemi]["min"/"max"] lists
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
from bk_embedding.mepOverlap import start_bokeh_app
from utils.tms_module import load_data


def save_mep_window_to_session(session_file: str, window_s: tuple[float, float]) -> None:
    beg, end = map(float, window_s)
    if beg >= end:
        raise ValueError(f"Invalid MEP window: {beg} >= {end}")

    js = json.loads(Path(session_file).read_text())
    js["mep_window"] = [beg, end]
    Path(session_file).write_text(json.dumps(js, indent=2))


def compute_and_store_minmax(session_file: str, meta: dict) -> None:
    """
    Uses:
      - session["meps"][block][hemi]["pulses"] (sample indices)
      - session["mep_window"] = [beg_s, end_s]
    Computes:
      - min/max inside [beg_s, end_s] relative to pulse
    """
    js = json.loads(Path(session_file).read_text())
    w = js.get("mep_window", [None, None])
    if not (isinstance(w, list) and len(w) == 2 and w[0] is not None and w[1] is not None):
        raise RuntimeError("Missing mep_window in session file.")

    beg_s, end_s = float(w[0]), float(w[1])
    fs = float(meta["sampling_rate"])

    # Load full raw data once
    data, _tms = load_data(meta["input_file"], channels=meta.get("channels"))
    hemis = list(meta.get("hemispheres", ["left"]))

    # Map hemi -> channel index in data returned by load_data()
    # tms_module.load_data returns [left, right] => left=0 right=1
    hemi_to_ch = {"left": 0, "right": 1}

    meps_root = js.get("meps", {})
    if not isinstance(meps_root, dict):
        return

    # also baseline-correct using pre-stim [-100ms, 0]
    pre_s = 0.100
    pre_n = int(round(pre_s * fs))

    for block, block_payload in meps_root.items():
        if not isinstance(block_payload, dict):
            continue

        for hemi in hemis:
            hp = block_payload.get(hemi, {})
            if not isinstance(hp, dict):
                continue

            pulses = hp.get("pulses", [])
            if not isinstance(pulses, list) or not pulses:
                hp["min"] = []
                hp["max"] = []
                continue

            ch = hemi_to_ch.get(hemi, 0)
            sig = data[ch, :]

            # indices for the mep window relative to pulse
            a_off = int(round(beg_s * fs))
            b_off = int(round(end_s * fs))
            if a_off >= b_off:
                raise ValueError("mep_window invalid after discretization (beg >= end).")

            mins, maxs = [], []
            for p in pulses:
                p = int(p)

                # baseline window: [p-pre_n, p)
                b0 = p - pre_n
                b1 = p
                if b0 < 0 or b1 <= 0 or b1 > sig.shape[0]:
                    baseline = 0.0
                else:
                    baseline = float(sig[b0:b1].mean())

                a = p + a_off
                b = p + b_off
                if a < 0 or b > sig.shape[0]:
                    mins.append(None)
                    maxs.append(None)
                    continue

                y = sig[a:b] - baseline
                mins.append(float(np.min(y)))
                maxs.append(float(np.max(y)))

            hp["min"] = mins
            hp["max"] = maxs

            # keep lists aligned if flags not present
            n = len(pulses)
            hp.setdefault("preactivation_flag", [0] * n)
            hp.setdefault("peaks_flag", [0] * n)
            if len(hp["preactivation_flag"]) != n:
                hp["preactivation_flag"] = (hp["preactivation_flag"] + [0] * n)[:n]
            if len(hp["peaks_flag"]) != n:
                hp["peaks_flag"] = (hp["peaks_flag"] + [0] * n)[:n]

            block_payload[hemi] = hp

        meps_root[block] = block_payload

    js["meps"] = meps_root
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

    def _on_next():
        with st.session_state["_ranges_lock"]:
            win = st.session_state["_ranges_store"].get("epoch_window")

        if not win:
            st.toast("Select a window before advancing", icon="⚠️")
            return False

        try:
            save_mep_window_to_session(session_file, win)
            compute_and_store_minmax(session_file, meta)
        except Exception as e:
            st.toast(f"Failed to save/compute min/max: {e}", icon="❌")
            return False

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
        heading_level=1,
        nowrap=True,
    )

    # Restart Bokeh if context changes
    bokeh_key = (session_file, meta.get("input_file"), meta.get("sampling_rate"))
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

    # iframe styling
    st.markdown(
        """
        <style>
        [data-testid="stAppViewBlockContainer"] { max-width: 100% !important; padding-left: 0rem !important; padding-right: 0rem !important; }
        section.main > div { max-width: 100% !important; padding-left: 0rem !important; padding-right: 0rem !important; }
        html, body, [data-testid="stAppViewContainer"] { overflow-x: hidden; }

        .bk-viewport-band { width: 92vw; position: relative; left: 50%; margin-left: -46vw; display: flex; justify-content: center; }
        .bk-viewport-band iframe { display: block; border: none; width: 92vw; height: 46vw; max-height: 65vh; }
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

    with st.session_state["_ranges_lock"]:
        epoch_window = st.session_state["_ranges_store"].get("epoch_window")
    st.caption(f"Current epoch window (s): {epoch_window}")

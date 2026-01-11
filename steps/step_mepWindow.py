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
import numpy as np
from scipy.signal import find_peaks
from math import ceil

from utils.persistence import (
    ensure_metadata,
    ensure_template_loaded,
    ensure_session_file,
)
from utils.layout import render_text, step_nav
from bk_embedding.mepOverlap import start_bokeh_app
from utils.tms_module import load_data, read_json, get_epoch_from_session

PREV_STEP = 'segmentation'
THIS_STEP = 'mep_window'
NEXT_STEP = "peak_checking"


def save_mep_window_to_session(session_file: str, window_s: tuple[float, float]) -> None:
    beg, end = map(float, window_s)
    if beg >= end:
        raise ValueError(f"Invalid MEP window: {beg} >= {end}")

    js = json.loads(Path(session_file).read_text())
    js["mep_window"] = [beg, end]
    Path(session_file).write_text(json.dumps(js, indent=2))

def _best_peak_idx_and_val(x: np.ndarray, pk_width: int | None, distance: int) -> tuple[int, float]:
    """
    Return (idx, value) for the most prominent peak in x.
    If none found, return (0, x[0]).
    """
    kwargs = {"distance": distance}
    if pk_width is not None and pk_width > 0:
        kwargs["width"] = pk_width

    peaks, props = find_peaks(x, **kwargs, prominence=0)  # one call
    if peaks.size == 0:
        return 0, float(x[0])

    # Choose the most prominent peak (robust if >1 slips through)
    prom = props.get("prominences")
    best_j = int(np.argmax(prom)) if prom is not None and len(prom) else 0
    idx = int(peaks[best_j])
    return idx, float(x[idx])


def compute_and_store_minmax(session_file: str, meta: dict) -> None:
    """
    Uses:
      - session["meps"][block][hemi]["pulses"] (sample indices)
      - epoch window from session (same source as step_peakChecking)
        (i.e., via get_epoch_from_session(session))

    Computes:
      - min/max inside [epoch.tmin_ms, epoch.tmax_ms] relative to pulse
        (converts ms -> samples using fs)
    """
    # --- load session + epoch exactly like step_peakChecking ---
    js = read_json(Path(session_file))
    epoch = get_epoch_from_session(js)

    fs = float(meta["sampling_rate"])

    # Load full raw data once
    data, _tms = load_data(meta["input_file"], channels=meta.get("channels"))
    hemis = list(meta.get("hemispheres", ["left"]))

    hemi_to_ch = {"left": 0, "right": 1}

    meps_root = js.get("meps", {})
    if not isinstance(meps_root, dict):
        return

    # baseline-correct using pre-stim [-100ms, 0]
    pre_s = 0.100
    pre_n = int(round(pre_s * fs))

    # epoch bounds in *samples* (epoch is in ms relative to pulse)
    a_off = int(round((epoch.tmin_ms / 1000.0) * fs))
    b_off = int(round((epoch.tmax_ms / 1000.0) * fs))
    if a_off >= b_off:
        raise ValueError("Epoch invalid after discretization (tmin >= tmax).")

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

            # find peaks
            pk_width = ceil(fs*0.001)

            mins, maxs = [], []
            mep_start = float(epoch.tmin_ms)
            dist = None  # set per MEP (depends on length)

            for p in pulses:
                p = int(p)
                a = p + a_off
                b = p + b_off

                if a < 0 or b > sig.shape[0]:
                    mins.append((None, None))
                    maxs.append((None, None))
                    continue

                mep = sig[a:b].astype(float)
                dist = max(1, len(mep) - 1)

                max_i, max_val = _best_peak_idx_and_val(mep, pk_width=pk_width, distance=dist)
                min_i, min_val = _best_peak_idx_and_val(-mep, pk_width=pk_width, distance=dist)
                min_val = -min_val  # because we ran on -mep

                max_ms = (max_i * 1000.0 / fs) + mep_start
                min_ms = (min_i * 1000.0 / fs) + mep_start

                maxs.append((float(max_ms), float(max_val)))
                mins.append((float(min_ms), float(min_val)))

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
        THIS_STEP,
        step_title = "MEP window definition",
        back_step=PREV_STEP,
        next_step=NEXT_STEP,
        on_next=_on_next,
        disabled_next=False,
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

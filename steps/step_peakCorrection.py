"""
steps/step_peakCorrection.py
-----------------------------
Step 6: Peak correction (with draggable points via embedded Dash)

- Iterate only MEPs with peaks_flag == 1 (across all *meps blocks)
- Drag min/max points in embedded Dash plot
- Dash writes [ms, val] into session JSON
- Streamlit handles queue navigation + reject + autosave for peaks_flag

Notes:
- Avoid components.html + JS URL construction because Streamlit component sandbox
  can block scripts/popups and you end up with a blank frame + dead links.
- Instead: build Dash URLs in Python and use components.iframe + Streamlit link_button.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import socket
import threading
import urllib.parse

import streamlit as st
import streamlit.components.v1 as components

from utils.dash_peak_editor import create_dash_peak_editor
from utils.persistence import (
    ensure_metadata,
    ensure_template_loaded,
    ensure_session_file,
)
from utils.layout import render_text, step_nav
from utils.tms_module import (
    Epoch,
    read_json,
    write_json,
    get_epoch_from_session,
)

PREV_STEP = "peak_checking"
THIS_STEP = "peak_correction"
NEXT_STEP = None


# -------------------------
# Queue item
# -------------------------
@dataclass(frozen=True)
class MEPKey:
    block: str
    hemi: str
    mep_idx: int


# -------------------------
# Dash boot helpers
# -------------------------
def _find_free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


class _StripFrameHeadersMiddleware:
    """Strip frame-blocking headers from Dash responses (local embedding)."""
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        def _start_response(status, headers, exc_info=None):
            blocked = {"x-frame-options", "content-security-policy", "frame-options"}
            headers2 = [(k, v) for (k, v) in headers if k.lower() not in blocked]
            return start_response(status, headers2, exc_info)

        return self.app(environ, _start_response)


def _dash_code_sig(meta: dict) -> float:
    """
    Signature used to restart Dash when dash editor code changes.
    Uses the mtime of utils/dash_peak_editor.py.

    NOTE: no global 'meta' access.
    """
    base = Path(meta.get("_script_dir", "."))
    p = (base / "utils" / "dash_peak_editor.py").resolve()
    try:
        return float(p.stat().st_mtime)
    except Exception:
        return 0.0


def _ensure_dash_running(meta: dict, session_file: str | Path) -> int:
    """
    Start Dash once per Streamlit session, unless dash editor code changed.
    """
    session_file = str(session_file)
    sig = _dash_code_sig(meta)

    # reuse existing if signature unchanged
    if (
        st.session_state.get("_pcorr_dash_started")
        and st.session_state.get("_pcorr_dash_port")
        and st.session_state.get("_pcorr_dash_sig") == sig
    ):
        return int(st.session_state["_pcorr_dash_port"])

    # otherwise start a fresh instance (new port)
    port = _find_free_port()
    dash_app = create_dash_peak_editor(meta=meta, session_file=session_file)

    # Strip frame-blocking headers if possible (helps embedding)
    try:
        dash_app.server.wsgi_app = _StripFrameHeadersMiddleware(dash_app.server.wsgi_app)
    except Exception:
        pass

    def _run():
        # Dash >= 2.15 prefers app.run()
        dash_app.run(host="0.0.0.0", port=port, debug=False)

    threading.Thread(target=_run, daemon=True).start()

    st.session_state["_pcorr_dash_started"] = True
    st.session_state["_pcorr_dash_port"] = int(port)
    st.session_state["_pcorr_dash_sig"] = sig
    return int(port)


def _reset_dash() -> None:
    """Force restart on next _ensure_dash_running()."""
    st.session_state.pop("_pcorr_dash_started", None)
    st.session_state.pop("_pcorr_dash_port", None)
    st.session_state.pop("_pcorr_dash_sig", None)


def _guess_streamlit_hostname() -> str:
    """
    Best-effort host detection without hardcoding.

    - If user set Streamlit config browser.serverAddress / server.address, use it.
    - Else fall back to localhost.
    """
    candidates = [
        st.get_option("browser.serverAddress"),
        st.get_option("server.address"),
    ]

    for c in candidates:
        if isinstance(c, str):
            c = c.strip()
            if c and c not in {"0.0.0.0", "127.0.0.1"}:
                return c

    return "localhost"


def _guess_streamlit_scheme() -> str:
    """
    Best-effort scheme detection.
    Streamlit local is typically http.
    """
    return "http"


def _dash_urls(port: int, *, block: str, hemi: str, mep_idx: int) -> tuple[str, str]:
    host = _guess_streamlit_hostname()
    scheme = _guess_streamlit_scheme()

    qs = urllib.parse.urlencode({"block": block, "hemi": hemi, "mep": str(int(mep_idx))})
    dash_url = f"{scheme}://{host}:{int(port)}/?{qs}"
    health_url = f"{scheme}://{host}:{int(port)}/health"
    return dash_url, health_url


# -------------------------
# Session JSON helpers (flag + queue)
# -------------------------
def _ensure_payload_lists(payload: dict) -> None:
    pulses = payload.get("pulses", [])
    n = len(pulses) if isinstance(pulses, list) else 0

    for k in ("min", "max", "peaks_flag"):
        if not isinstance(payload.get(k), list):
            payload[k] = []

    def _fit(name: str, fill):
        arr = payload[name]
        if len(arr) < n:
            arr.extend([fill] * (n - len(arr)))
        elif len(arr) > n:
            payload[name] = arr[:n]

    _fit("peaks_flag", 0)
    _fit("min", None)
    _fit("max", None)


def _get_payload(session: dict, block: str, hemi: str) -> dict:
    meps = session.get("meps", {}) or {}
    payload = ((meps.get(str(block)) or {}).get(str(hemi)) or {})
    if not isinstance(payload, dict):
        payload = {}
    _ensure_payload_lists(payload)
    return payload


def _set_peaks_flag(session: dict, key: MEPKey, value: int) -> None:
    session.setdefault("meps", {}).setdefault(key.block, {}).setdefault(key.hemi, {})
    payload = session["meps"][key.block][key.hemi]
    if not isinstance(payload, dict):
        payload = {}
    _ensure_payload_lists(payload)

    payload["peaks_flag"][key.mep_idx] = 1 if int(value) else 0
    session["meps"][key.block][key.hemi] = payload


def _build_queue(session: dict, blocks: list[str], hemi: str) -> list[MEPKey]:
    out: list[MEPKey] = []
    for b in blocks:
        payload = _get_payload(session, b, hemi)
        flags = payload.get("peaks_flag", []) or []
        for i, f in enumerate(flags):
            try:
                if int(f) == 1:
                    out.append(MEPKey(block=str(b), hemi=str(hemi), mep_idx=int(i)))
            except Exception:
                continue
    return out


# -------------------------
# State keys
# -------------------------
def _q_key() -> str:
    return "_pcorr_queue"

def _pos_key() -> str:
    return "_pcorr_pos"

def _autosave_key() -> str:
    return "_pcorr_autosave"

def _hemi_key() -> str:
    return "_pcorr_hemi"


# -------------------------
# Step
# -------------------------
def run_step(meta: dict):
    meta = ensure_metadata()
    meta = ensure_template_loaded(meta)
    session_file = Path(ensure_session_file(meta))

    step_nav(
        THIS_STEP,
        step_title="Peak Correction",
        next_step=NEXT_STEP,
        back_step=PREV_STEP,
        disabled_next=True,
    )

    session = read_json(session_file)
    _epoch: Epoch = get_epoch_from_session(session)

    blocks = [b for b in (meta.get("exp_structure", []) or []) if str(b).lower().endswith("meps")]
    if not blocks:
        st.error("No '*meps' blocks found in meta['exp_structure'].")
        return

    hemis = list((session.get("info", {}) or {}).get("hemispheres", meta.get("hemispheres", ["left"])))
    if not hemis:
        hemis = ["left"]

    if _hemi_key() not in st.session_state:
        st.session_state[_hemi_key()] = str(hemis[0])

    topL, topR, topX = st.columns([1, 1, 1])

    with topL:
        hemi = st.selectbox(
            "Hemisphere",
            hemis,
            index=hemis.index(st.session_state[_hemi_key()]) if st.session_state[_hemi_key()] in hemis else 0,
            key="pcorr_hemi_sel",
        )
        st.session_state[_hemi_key()] = str(hemi)

    with topR:
        if _autosave_key() not in st.session_state:
            st.session_state[_autosave_key()] = True
        st.session_state[_autosave_key()] = st.checkbox(
            "Autosave on Prev/Next (flag only)",
            value=bool(st.session_state[_autosave_key()]),
            key="pcorr_autosave_chk",
        )

    with topX:
        if st.button("🧨 Restart Dash editor", use_container_width=True):
            _reset_dash()
            st.rerun()

    if _q_key() not in st.session_state or st.button("🔄 Rebuild queue (peaks_flag==1)", use_container_width=True):
        session = read_json(session_file)
        q = _build_queue(session, blocks=[str(b) for b in blocks], hemi=str(hemi))
        st.session_state[_q_key()] = [qq.__dict__ for qq in q]
        st.session_state[_pos_key()] = 0

    queue = st.session_state.get(_q_key(), [])
    if not queue:
        st.info("No MEPs with peaks_flag == 1. Nothing to correct.")
        return

    if _pos_key() not in st.session_state:
        st.session_state[_pos_key()] = 0

    pos = int(st.session_state[_pos_key()])
    pos = max(0, min(pos, len(queue) - 1))
    st.session_state[_pos_key()] = pos

    key = MEPKey(**queue[pos])

    session = read_json(session_file)
    payload = _get_payload(session, key.block, key.hemi)
    stored_flag = int(payload["peaks_flag"][key.mep_idx]) if key.mep_idx < len(payload["peaks_flag"]) else 0

    render_text(
        f"{pos+1}/{len(queue)}  |  block={key.block}  hemi={key.hemi}  mep_idx={key.mep_idx}  |  peaks_flag={stored_flag}",
        horizontal_alignment="center",
        nowrap=False,
    )

    reject = st.checkbox("Reject this MEP (set peaks_flag=0)", value=False, key="pcorr_reject")

    # --- Draggable editor (Dash) ---
    port = _ensure_dash_running(meta, session_file)
    dash_url, health_url = _dash_urls(port, block=key.block, hemi=key.hemi, mep_idx=key.mep_idx)

    # Real Streamlit links (not sandboxed)
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.link_button("Open Dash directly", dash_url, use_container_width=True)
    with c2:
        st.link_button("Health", health_url, use_container_width=True)
    with c3:
        st.caption(f"Dash URL: {dash_url}")

    # Embed Dash
    # --- Centered, responsive, non-scrollable iframe (CSS-driven) ---
    st.markdown(
        """
        <style>
        /* wrapper centers content and prevents horizontal scroll */
        .dash-embed-wrap {
            width: 100%;
            display: flex;
            justify-content: center;
            overflow: hidden;
        }

        /* iframe scales with window size */
        .dash-embed-frame {
            width: min(1200px, 100%);
            height: 55vh;           /* responsive height */
            border: 0;
            overflow: hidden;
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="dash-embed-wrap">
        <iframe
            class="dash-embed-frame"
            src="{dash_url}"
            scrolling="no"
            sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
        ></iframe>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # --- Nav / save peaks_flag ---
    b1, b2, b3, b4 = st.columns([1, 1, 1, 1])

    def _save_flag_only() -> None:
        sess = read_json(session_file)
        _set_peaks_flag(sess, key, 0 if reject else 1)
        write_json(session_file, sess)

    with b1:
        if st.button("💾 Save flag", use_container_width=True):
            _save_flag_only()
            st.toast("Saved peaks_flag.", icon="💾")

    with b2:
        if st.button("◀ Prev", use_container_width=True, disabled=(pos == 0)):
            if st.session_state[_autosave_key()]:
                _save_flag_only()
            st.session_state[_pos_key()] = max(0, pos - 1)
            st.rerun()

    is_last = (pos >= len(queue) - 1)

    with b3:
        label = "Finish ✅" if is_last else "Next ▶"
        if st.button(label, use_container_width=True):
            # Still save reject/flag if you want that behavior
            if st.session_state[_autosave_key()]:
                _save_flag_only()

            if is_last:
                # done: optional cleanup / message
                st.success("Finished peak correction for all flagged MEPs.")
                # Optionally clear queue so user doesn't re-enter mid-way:
                # st.session_state.pop(_q_key(), None)
                # st.session_state.pop(_pos_key(), None)
                st.stop()
            else:
                st.session_state[_pos_key()] = min(len(queue) - 1, pos + 1)
                st.rerun()


    with b4:
        if st.button("✅ Save flag + Rebuild queue", use_container_width=True):
            _save_flag_only()
            sess = read_json(session_file)
            q = _build_queue(sess, blocks=[str(b) for b in blocks], hemi=str(hemi))
            st.session_state[_q_key()] = [qq.__dict__ for qq in q]
            st.session_state[_pos_key()] = min(st.session_state[_pos_key()], max(0, len(q) - 1))
            st.rerun()

    st.caption(
        "Min/max are saved from the embedded Dash plot via its **Save min/max** button.\n"
        "This Streamlit step only manages peaks_flag + navigation."
    )

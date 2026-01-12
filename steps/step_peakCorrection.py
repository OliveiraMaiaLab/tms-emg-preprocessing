"""
steps/step_peakCorrection.py
-----------------------------
Step 6: Peak correction (with draggable points via embedded Dash)

- Iterate only MEPs with peaks_flag == 1 (across all *meps blocks AND all hemispheres)
- Drag min/max points in embedded Dash plot
- Dash writes [ms, val] into session JSON (min/max only)
- Streamlit rebuilds the queue automatically from session JSON on every rerun
- Streamlit tracks which flagged MEPs have already been reviewed (visited) so they
  do NOT reappear even if the queue is rebuilt.

Notes:
- This step does NOT modify peaks_flag (or any other flags).
- "Finish ✅" (top-right) is enabled only when the remaining queue is empty
  OR when you're on the last remaining item (depending on the logic below).
  Here: enabled only when you're on the last remaining item.
- On Finish: update processed_sessions.json + show flash on Step 1 + jump to input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import socket
import threading
import urllib.parse

import streamlit as st

from utils.dash_peak_editor import create_dash_peak_editor, SHOW_DEBUG
from utils.persistence import (
    ensure_metadata,
    ensure_template_loaded,
    ensure_session_file,
    update_processed_sessions_registry,
)
from utils.layout import render_text, step_nav
from utils.tms_module import read_json

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
    """Restart Dash when utils/dash_peak_editor.py changes."""
    base = Path(meta.get("_script_dir", "."))
    p = (base / "utils" / "dash_peak_editor.py").resolve()
    try:
        return float(p.stat().st_mtime)
    except Exception:
        return 0.0


def _ensure_dash_running(meta: dict, session_file: str | Path) -> int:
    """Start Dash once per Streamlit session, unless dash editor code changed."""
    session_file = str(session_file)
    sig = _dash_code_sig(meta)

    if (
        st.session_state.get("_pcorr_dash_started")
        and st.session_state.get("_pcorr_dash_port")
        and st.session_state.get("_pcorr_dash_sig") == sig
    ):
        return int(st.session_state["_pcorr_dash_port"])

    port = _find_free_port()
    dash_app = create_dash_peak_editor(meta=meta, session_file=session_file)

    try:
        dash_app.server.wsgi_app = _StripFrameHeadersMiddleware(dash_app.server.wsgi_app)
    except Exception:
        pass

    def _run():
        dash_app.run(host="0.0.0.0", port=port, debug=False)

    threading.Thread(target=_run, daemon=True).start()

    st.session_state["_pcorr_dash_started"] = True
    st.session_state["_pcorr_dash_port"] = int(port)
    st.session_state["_pcorr_dash_sig"] = sig
    return int(port)


def _guess_streamlit_hostname() -> str:
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
    return "http"


def _dash_url(port: int, *, block: str, hemi: str, mep_idx: int) -> str:
    host = _guess_streamlit_hostname()
    scheme = _guess_streamlit_scheme()
    qs = urllib.parse.urlencode({"block": block, "hemi": hemi, "mep": str(int(mep_idx))})
    return f"{scheme}://{host}:{int(port)}/?{qs}"


# -------------------------
# Session JSON helpers (queue only)
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


def _build_queue(session: dict, blocks: list[str], hemis: list[str]) -> list[MEPKey]:
    """Build queue across ALL hemispheres and blocks: include items where peaks_flag[i] == 1."""
    out: list[MEPKey] = []
    for b in blocks:
        for h in hemis:
            payload = _get_payload(session, b, h)
            flags = payload.get("peaks_flag", []) or []
            for i, f in enumerate(flags):
                try:
                    if int(f) == 1:
                        out.append(MEPKey(block=str(b), hemi=str(h), mep_idx=int(i)))
                except Exception:
                    continue
    return out


# -------------------------
# Visited tracking
# -------------------------
def _visited_key() -> str:
    return "_pcorr_visited"  # set[str] of "block|hemi|idx"


def _current_key() -> str:
    return "_pcorr_current"  # "block|hemi|idx" or ""


def _key_str(k: MEPKey) -> str:
    return f"{k.block}|{k.hemi}|{k.mep_idx}"


def _parse_key_str(s: str) -> MEPKey | None:
    try:
        b, h, i = s.split("|")
        return MEPKey(block=b, hemi=h, mep_idx=int(i))
    except Exception:
        return None


def _remaining_queue(session: dict, blocks: list[str], hemis: list[str], visited: set[str]) -> list[MEPKey]:
    fresh = _build_queue(session, blocks=blocks, hemis=hemis)
    # Filter out already visited
    return [k for k in fresh if _key_str(k) not in visited]


# -------------------------
# Finish logic
# -------------------------
def _finish_to_input(meta: dict, session_file: Path) -> None:
    update_processed_sessions_registry(
        output_dir=meta.get("output_dir") or session_file.parent,
        data_file=meta.get("input_file", "UNKNOWN_DATA_FILE"),
        session_file=session_file,
        researcher_id=meta.get("researcher_id", ""),
        pipeline_version=meta.get("version", ""),
    )

    st.session_state["_global_flash_success"] = f'Finished processing "{session_file.name}".'
    st.session_state.step = "input"
    st.rerun()


# -------------------------
# Step
# -------------------------
def run_step(meta: dict):
    meta = ensure_metadata()
    meta = ensure_template_loaded(meta)
    session_file = Path(ensure_session_file(meta))

    # Init visited/current
    if _visited_key() not in st.session_state:
        st.session_state[_visited_key()] = set()
    visited: set[str] = st.session_state[_visited_key()]

    # Read session once per rerun (queue is rebuilt from this)
    session = read_json(session_file)

    blocks = [b for b in (meta.get("exp_structure", []) or []) if str(b).lower().endswith("meps")]
    if not blocks:
        st.error("No '*meps' blocks found in meta['exp_structure'].")
        return

    hemis = list((session.get("info", {}) or {}).get("hemispheres", meta.get("hemispheres", ["left"])))
    if not hemis:
        hemis = ["left"]

    # Rebuild remaining queue every run (and filter visited)
    remaining = _remaining_queue(
        session,
        blocks=[str(b) for b in blocks],
        hemis=[str(h) for h in hemis],
        visited=visited,
    )

    # If nothing left, enable Finish immediately
    if not remaining:
        step_nav(
            THIS_STEP,
            step_title="Peak Correction",
            right_label="Finish ✅",
            next_step=NEXT_STEP,          # None
            back_step=PREV_STEP,
            disabled_next=False,          # enabled
            on_next=(lambda: _finish_to_input(meta, session_file)),
        )
        st.success("No remaining flagged MEPs to correct.")
        return

    # Pick current item (keep it if still present in remaining)
    cur_s = st.session_state.get(_current_key(), "") or ""
    cur = _parse_key_str(cur_s)

    remaining_set = {_key_str(k) for k in remaining}
    if cur is None or _key_str(cur) not in remaining_set:
        cur = remaining[0]
        st.session_state[_current_key()] = _key_str(cur)

    # Position of current in remaining
    pos = next((i for i, k in enumerate(remaining) if _key_str(k) == _key_str(cur)), 0)
    key = remaining[pos]
    is_last = (pos >= len(remaining) - 1)

    # Top nav: Finish enabled only on last remaining item
    step_nav(
        THIS_STEP,
        step_title="Peak Correction",
        right_label="Finish ✅",
        next_step=NEXT_STEP,              # None
        back_step=PREV_STEP,
        disabled_next=(not is_last),
        on_next=(lambda: _finish_to_input(meta, session_file)) if is_last else None,
    )

    render_text(
        f"MEP {key.mep_idx} in&nbsp;<em>{key.block}</em>",
        horizontal_alignment="center",
        nowrap=True,
        heading_level=3,
    )

    # --- Draggable editor (Dash) ---
    port = _ensure_dash_running(meta, session_file)
    dash_url = _dash_url(port, block=key.block, hemi=key.hemi, mep_idx=key.mep_idx)

    if SHOW_DEBUG:
        st.link_button("Open Dash directly", dash_url, use_container_width=True)

    # --- Centered, responsive, non-scrollable iframe ---
    st.markdown(
        """
        <style>
        .dash-embed-wrap {
            width: 100%;
            display: flex;
            justify-content: center;
            overflow: hidden;
        }
        .dash-embed-frame {
            width: min(1200px, 100%);
            height: 55vh;
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

    # --- Navigation ---
    b1, b2, b3 = st.columns([1, 1, 1])

    with b1:
        if st.button("◀ Prev", use_container_width=True, disabled=(pos == 0)):
            prev_key = remaining[pos - 1]
            st.session_state[_current_key()] = _key_str(prev_key)
            st.rerun()

    with b2:
        render_text(
            f"MEP {pos+1}/{len(remaining)}  ({key.hemi} hemisphere)",
            horizontal_alignment="center",
            nowrap=True,
        )

    with b3:
        # "Next" marks current as visited so it won't reappear even if queue rebuilds
        if st.button("Next ▶", use_container_width=True, disabled=is_last):
            visited.add(_key_str(key))
            st.session_state[_visited_key()] = visited

            # Recompute remaining after marking visited
            session2 = read_json(session_file)
            remaining2 = _remaining_queue(
                session2,
                blocks=[str(b) for b in blocks],
                hemis=[str(h) for h in hemis],
                visited=visited,
            )

            if not remaining2:
                # Nothing left → allow Finish
                st.session_state[_current_key()] = ""
                st.rerun()

            # keep same forward-ish position (clamped)
            next_pos = min(pos, len(remaining2) - 1)
            st.session_state[_current_key()] = _key_str(remaining2[next_pos])
            st.rerun()
